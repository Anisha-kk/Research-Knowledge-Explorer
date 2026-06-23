import json
import os

class StateManager:
    def __init__(self):
        self.path = "data/state.json"
        os.makedirs("data", exist_ok=True)

        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump({}, f)

    def load(self):
        with open(self.path, "r") as f:
            return json.load(f)

    def save(self, state):
        with open(self.path, "w") as f:
            json.dump(state, f, indent=2)

    def is_done(self, title):#Is the title already present in the state.json file
        state = self.load()
        return title in state

    def mark_done(self, title, data):#Add data to the given title
        state = self.load()
        state[title] = data
        self.save(state)

