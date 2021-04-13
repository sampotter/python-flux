import glob
import os
from plapp.config import FluxOpt
from plapp.proc_objfil import main as proc_opjfil
import logging

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    FluxOpt.set('body', 'MOON')
    FluxOpt.set('new_obj', True)
    FluxOpt.set('new_FF', True)
    FluxOpt.set('use_spice', True)
    FluxOpt.set("spice_utc_start", '2021 FEB 15 00:00:00.00')
    FluxOpt.set("spice_utc_end", '2021 FEB 18 00:00:00.00')
    FluxOpt.set("spice_utc_stepet", 24 * 3600)
    FluxOpt.set("use_distant_topo", True)
    FluxOpt.set("tree_kind", "quad")

    # Always append after changing options, to check if options make sense
    FluxOpt.check_consistency()

    # get example dir
    FluxOpt.set("example_dir", os.getcwd())

    # set obj file to import
    if FluxOpt.get('body') == 'MOON':
        flist = [f'LDEM_75S_240M_Haworth.grd']
        FluxOpt.set("resolution", 4.) # km

    print('flist:',flist)
    for filein in flist:
        print("Processing", FluxOpt.get('body'), " (obj file from", filein, ") ...")
        proc_opjfil(filein=filein)