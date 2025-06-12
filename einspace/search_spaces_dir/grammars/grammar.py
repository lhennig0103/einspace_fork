import random

class GrammarSearchSpace:
    def __init__(self, grammar_file):
        self.grammar = self._load_grammar(grammar_file)

    def _load_grammar(self, grammar_file):
        grammar = {}
        with open(grammar_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                lhs, rhs = line.split('::=')
                lhs = lhs.strip().strip('<>')
                rhs_options = [option.strip() for option in rhs.split('|')]
                grammar[lhs] = rhs_options
        return grammar

    def sample(self):
        return self._generate('start')

    def _generate(self, symbol):
        if symbol not in self.grammar:
            return symbol
        production = random.choice(self.grammar[symbol])
        tokens = production.split()
        return ' '.join([self._generate(token.strip('<>')) if token.startswith('<') and token.endswith('>') else token for token in tokens])

    def decode(self, genotype):
        return genotype
