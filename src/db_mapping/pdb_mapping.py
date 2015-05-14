'''
Mapping properties from PDB structures to Uniprot proteins using SIFTS data.
'''
import os.path
from lxml import etree
import urllib
import prody
from collections import defaultdict
from itertools import izip

class NoSIFTSMappingException(Exception):
    pass

SIFTS_URL = "ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/xml/{0}.xml.gz"

def map_atom_group(atoms, pdb_id =None):

    if pdb_id is None:
        pdb_id = atoms.getTitle()

    pdb_id = pdb_id.lower()

    url = SIFTS_URL.format(pdb_id)
    filename = url.split('/')[-1]

    if not os.path.isfile(filename):
        try:
            urllib.urlretrieve(url, filename)
        except IOError:
            raise NoSIFTSMappingException()

    try:
        d = etree.parse(filename)
    except etree.XMLSyntaxError:
        raise NoSIFTSMappingException()
    root = d.getroot()

    ns = {"xmlns" :"http://www.ebi.ac.uk/pdbe/docs/sifts/eFamily.xsd"}

    chains = defaultdict(dict)
    for residue in root.iterfind(".//xmlns:residue", namespaces=ns):
        pdb_ref = residue.find("xmlns:crossRefDb[@dbSource='PDB']", namespaces=ns)
        uniprot_ref = residue.find("xmlns:crossRefDb[@dbSource='UniProt']", namespaces=ns)
        if pdb_ref is not None and uniprot_ref is not None:
            chains[pdb_ref.get('dbChainId')][pdb_ref.get('dbResNum')] = (uniprot_ref.get('dbAccessionId'), int(uniprot_ref.get('dbResNum')))

    uids = []
    uresnums = []

    for a in atoms:
        try:
            v = chains[a.getChid()][str(a.getResnum())+a.getIcode()]
            uids.append(v[0])
            uresnums.append(v[1])
        except KeyError:
            uids.append(None)
            uresnums.append(None)

    atoms.setData('uids', uids)
    atoms.setData('uresnums', uresnums)

    return uids, uresnums

def extract_secondary_structures(pdb_id, uid2data):

    atoms = prody.performDSSP(pdb_id)
    map_atom_group(atoms)
    uid2structure = {}
    for uid in uid2data:
        uid2structure[uid] = ["_"] * len(uid2data[uid]['sequence'])

    for uid, uresnum, secstr in izip(atoms.getData('uids'), atoms.getData('uresnums'), atoms.getSecstrs()):
        if uid is None:
            continue
        if secstr == "":
            secstr = "-"
        uid2structure[uid][uresnum-1] = secstr

    uid2structure = dict((uid, "".join(seq)) for uid, seq in uid2structure.iteritems())

    return uid2structure


def extract_acc(pdb_id):

    atoms = prody.performDSSP(pdb_id)
    map_atom_group(atoms)
    uid2acc = {}

    for uid, uresnum, acc in izip(atoms.getData('uids'),
                                  atoms.getData('uresnums'),
                                  atoms.getData('dssp_acc')):
        if uid is None:
            continue
        if uid not in uid2acc:
            uid2acc[uid] = []
        if uresnum-1 >= len(uid2acc[uid]):
            uid2acc[uid] += [-1]*(uresnum-len(uid2acc[uid]))
        uid2acc[uid][uresnum-1] = acc

    return uid2acc



if __name__ == "__main__":

    atoms = prody.performDSSP('104l')

    uids, uresnums = map_atom_group(atoms)

    for res in atoms.iterResidues():
        print(res.getResname(), res.getData('uids')[0], res.getData('uresnums')[0], res.getSecstrs()[0])

    structures = extract_secondary_structures('104l')
    print(structures)
