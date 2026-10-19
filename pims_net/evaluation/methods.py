"""Four explicitly defined methods sharing frozen L0 features and first-click SAM."""

import torch


class EVSAMSession:
    """Official frozen EV-SAM with cumulative points and prior low-resolution logits."""

    def __init__(self, initializer, features, sizes):
        self.initializer, self.features, self.sizes = initializer, features, sizes
        self.points, self.labels = [], []
        self.previous_logits = None

    def _prompt(self, x, y, label):
        self.points.append([x, y])
        self.labels.append(label)
        return (torch.tensor([self.points], dtype=torch.float32, device=self.features.device),
                torch.tensor([self.labels], device=self.features.device))

    def _original_probabilities(self, logits):
        return self.initializer.postprocess(logits, self.sizes)[0].sigmoid()[0, 0].cpu().numpy()

    @torch.no_grad()
    def add_click(self, x, y, label):
        points, labels = self._prompt(x, y, label)
        logits = self.initializer(self.features, points, labels, self.sizes,
                                  mask_input=self.previous_logits)
        self.previous_logits = logits
        return self._original_probabilities(logits)

    @torch.no_grad()
    def diagnostic_snapshot(self):
        return {"logits": self.previous_logits[0].cpu().numpy(),
                "probabilities": self.previous_logits[0].sigmoid().cpu().numpy(),
                "previous_probabilities": None, "intents": None, "original_intents": None}


class PIMSSession:
    """Run the same variant definition used during independent training."""

    def __init__(self, model, features, sizes):
        self.model, self.features, self.sizes = model, features, sizes
        self.state = None
        self.previous_probabilities = None

    @torch.no_grad()
    def add_click(self, x, y, label):
        if self.state is None:
            self.state = self.model.initial_state(self.features, self.sizes, [[[x, y]]], [[label]])
        else:
            self.previous_probabilities = self.state.probabilities
            self.state = self.model.refine(self.state, [[[x, y]]], [[label]])
        return self.model.sam_initializer.postprocess(self.state.logits, self.sizes)[0].sigmoid()[0, 0].cpu().numpy()

    @torch.no_grad()
    def diagnostic_snapshot(self):
        intents = self.state.edit_probabilities
        # Intent channels are already probabilities. Use only SAM's linear
        # resize/crop geometry; applying sigmoid here would corrupt heatmaps.
        original = None if intents is None else self.model.sam_initializer.postprocess(intents, self.sizes)[0][0].cpu().numpy()
        array = lambda value: None if value is None else value[0].detach().cpu().numpy()
        return {"logits": array(self.state.logits), "probabilities": array(self.state.probabilities),
                "previous_probabilities": array(self.previous_probabilities),
                "intents": array(intents), "original_intents": original}


def create_sessions(models, features, sizes):
    if set(models) != {"a1", "a2", "a3"} or any(model.variant != name for name, model in models.items()):
        raise ValueError("Expected three independently trained A1/A2/A3 model variants")
    return {
        "a0": EVSAMSession(models["a3"].sam_initializer, features, sizes),
        **{name: PIMSSession(model, features, sizes) for name, model in models.items()},
    }
