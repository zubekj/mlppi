import sys
import os
import tempfile
import shutil
import sqlite3
import urllib
import prody
from xml.etree.ElementTree import ParseError

from db_mapping import pdb_entry
from db_mapping.pdb_mapping import NoSIFTSMappingException

def create_protein_database(idsfile, dbfile):
    prody.proteins.wwpdb.wwPDBServer('EU')

    conn = sqlite3.connect(dbfile)
    cur = conn.cursor()

    db_builder = pdb_entry.DBBuilder()

    cur.execute('''SELECT name FROM sqlite_master WHERE type="table" AND
                      name="protein-protein"''')
    if not cur.fetchone():
        db_builder.create_protein_table(cur)

    cur.execute('''SELECT name FROM sqlite_master WHERE type="table" AND
                      name="protein-ligand"''')
    if not cur.fetchone():
        db_builder.create_ligand_table(cur)

    cur.execute('''SELECT name FROM sqlite_master WHERE type="table" AND
                      name="ptm"''')
    if not cur.fetchone():
        db_builder.create_ptm_table(cur)

    f = open(idsfile)
    cdir = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/db_mapping/pdb_data')

    for line in f:
        pdb_id = line.rstrip('\n')
        try:
            for e in db_builder.pp_from_pdb_id(pdb_id):
                db_builder.add_pp_entry(cur, e)
            #for e in db_builder.pl_from_pdb_id(pdb_id):
            #    db_builder.add_pl_entry(cur, e)
            #for e in db_builder.ptm_from_pdb_id(pdb_id):
            #    db_builder.add_ptm_entry(cur, e)
        except NoSIFTSMappingException:
            print("Skipping {0} -- no SIFTS mapping.".format(pdb_id))
        except ParseError:
            print("Skipping {0} -- outdated UniProt refs.".format(pdb_id))
        except TypeError:
            print("Skipping {0} -- DSSP failed.".format(pdb_id))
        conn.commit()

    os.chdir(cdir)
    f.close()
    db_builder.close()
    conn.close()

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print "Usage: {0} pdb_id_list dbname".format(sys.argv[0])
        exit(1)

    idsfile = sys.argv[1]
    dbfile = sys.argv[2]

    create_protein_database(idsfile, dbfile)
