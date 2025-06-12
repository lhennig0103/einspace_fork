from einspace.search_spaces.grammars.grammar import GrammarSearchSpace

class MinimalSearchSpace(GrammarSearchSpace):
    def __init__(self):
        super().__init__(grammar_file="einspace/search_spaces/grammars/minimal.cfg")

