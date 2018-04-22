class DocVector:
    """VSM"""
    def __init__(self, doc, label):
        terms = sorted(doc.split(' '))
        self.terms = sorted(set(terms))
        self.tf = {t: terms.count(t) for t in self.terms} # term frequency
        self.label = label

    def get_tf(self, term):
        """term frequency"""
        return self.tf.get(term, 0)