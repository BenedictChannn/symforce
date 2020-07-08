    # --------------------------------------------------------------------------
    # Handwritten methods for Pose2
    # These will get included into the autogenerated class header.
    # --------------------------------------------------------------------------

    def __init__(self, R=None, t=None):
        rotation = R if R is not None else Rot2()
        position = t if R is not None else [0., 0.]
        assert isinstance(rotation, Rot2)

        self.data = rotation.to_storage() + list(position)

    def rotation(self):
        return Rot2.from_storage(self.data[0:2])

    def position(self):
        return np.array(self.data[2:4])