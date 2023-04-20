class MetaMethod:
    '''
    Handles the actor weight updates manually
    '''
    def __init__(self, actor, writer, *args):
        self.actor = actor
        self.writer = writer


class Reptile(MetaMethod):
    '''
    Environments all have the same goal, which is selected each step
    '''
    def __init__(self, *args):
        super().__init__(*args)


class ParallelReptile(MetaMethod):
    '''
    Environments have random goals distributed among them
    This is similar to my interleaving idea
    '''
    def __init__(self, *args):
        super().__init__(*args)
