# Options configuration for flux applications
# TODO add setters and getters

class FluxOpt:

    _conf = {
        # env opt
        "debug": False,
        "pgda": False,  # True #
        "body" : 'MOON',
        "example_dir": "../examples/",

        # processing opt
        "new_FF" : True,
        "tree_kind" : 'quad', #'oct'

        # spice opt
        "use_spice" : True,
        "frame" : 'MOON_ME',
        "spice_utc_start": '2011 MAR 01 00:00:00.00',
        "spice_utc_end": '2011 MAR 02 00:00:00.00',
        "spice_utc_stepet": 3*3600,

        # craters opt
        "new_obj": True,
        "resolution": 3, # km
        "is_crater": True,
        "feature" : 'haworth',
        "use_distant_topo" : False, #

        # plotting opt
        "DO_3D_PLOTTING" : False
    }
    _setters = list(_conf.keys()) # ["body","frame","tree_kind","use_distant_topo","new_FF","use_spice","is_crater","feature"]

    @staticmethod
    def check_consistency():

        if FluxOpt._conf['body'] == 'MOON':
            FluxOpt.set("is_crater", True)
        else:
            FluxOpt.set("is_crater", False)

        # if FluxOpt.__conf['is_crater']:
        #     FluxOpt.set("tree_kind", "quad")
        # else:
        #     FluxOpt.set("feature", None)
        #     FluxOpt.set("use_distant_topo", False)
        #     FluxOpt.set("tree_kind", "oct")

        if FluxOpt._conf['new_obj']:
            FluxOpt.set("new_FF", True)

    @staticmethod
    def get(name):
        return FluxOpt._conf[name]

    @staticmethod
    def set(name, value):
        if name in FluxOpt._setters:
            FluxOpt._conf[name] = value
            print(f"### FluxOpt.{name} updated to {value}.")
        else:
            raise NameError("Name not accepted in set() method")

# example, suppose importing
# from config import Options
if __name__ == '__main__':

    print(FluxOpt.get("tree_kind"))

    opt = FluxOpt()
    print(opt.get("tree_kind"))

    # example of how to change body, frame, etc...
    print(opt.get("body"))
    opt.set("body","BENNU")
    print(opt.get("body"))
    opt.set("frame","IAU_BENNU")
    # I cannot set this option (also, I don't know how to
    # update dependent opts without reinitialising)
    opt.set("is_crater",False)
    print(opt.get("tree_kind"))