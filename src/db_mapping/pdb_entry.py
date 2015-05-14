import os
import sqlite3
import simplejson as json
from prody import *

import pdb_mapping
import up_parsing

MIN_PROTEIN_LEN = 50
PROT_INTERACT_DIST = 4
LIG_INTERACT_DIST = 3

class DBBuilder(object):

    def __init__(self):
        pathPDBFolder(os.path.dirname(os.path.abspath(__file__)) + '/pdb_data')
        self.up_parser = up_parsing.UPParser()


    def get_interactions_list(self, pdb_struct):
        connected_residues = set()
        hv = pdb_struct.getHierView()
        processed = set()
        for chain_A in hv:
            for chain_B in hv:
                if (chain_B, chain_A) in processed:
                    continue
                if chain_A != chain_B:
                    connected_residues.update(iterNeighbors(chain_A, PROT_INTERACT_DIST, chain_B))
                processed.add((chain_A, chain_B))
        return connected_residues


    def map_interactions(self, pdb_id, interactions_list):

        def getData(atom, label):
            return atom._ag._getData(label)[atom._index]

        up_interactions = {}
        for pair in interactions_list:
            u1, c1, r1 = getData(pair[0], 'uids'), pair[0].getChid(), getData(pair[0], 'uresnums')
            u2, c2, r2 = getData(pair[1], 'uids'), pair[1].getChid(), getData(pair[1], 'uresnums')

            if not (u1 and u2):
                #print "Skipping unmapped residues pair."
                continue
            if ((u2, c2), (u1, c1)) in up_interactions:
                up_interactions[((u2, c2), (u1, c1))].add((r2, r1))
                continue
            if not ((u1, c1), (u2, c2)) in up_interactions:
                up_interactions[((u1, c1), (u2, c2))] = set()
            up_interactions[((u1, c1), (u2, c2))].add((r1, r2))
        return up_interactions

    def get_up_data(self, uid_list):
        uid2data = {}
        for uid in uid_list:
            uid2data[uid] = self.up_parser.get_uniprot_entry(uid)
        return uid2data

    def get_ligand_interactions(self, id, chemical_resname, pdb_struct):

        def getData(atom, label):
            return atom._ag._getData(label)[atom._index]

        ch_atoms = pdb_struct.select('resname {0}'.format(chemical_resname))
        if not ch_atoms:
            print("Skipping chemical with the name {0}".format(chemical_resname))
            return []
        ligand_interactions = {}
        for pair in iterNeighbors(ch_atoms, LIG_INTERACT_DIST, pdb_struct):
            u1, r1 = getData(pair[0], 'uids'), getData(pair[0], 'uresnums')
            if not u1:
                #print "Skipping unmapped residue."
                continue
            if not u1 in ligand_interactions:
                ligand_interactions[u1] = set()
            ligand_interactions[u1].add(r1)
        return ligand_interactions

    def pp_from_pdb_id(self, pdb_id):
        """Gathers data from a single PDB pdb_id."""

        pdb_struct = parsePDB(pdb_id)

        reference = parsePDBHeader(pdb_id, 'reference')
        pmid = int(reference['pmid']) if 'pmid' in reference else 0

        pdb_mapping.map_atom_group(pdb_struct)

        up_interactions = self.map_interactions(pdb_id,
                self.get_interactions_list(pdb_struct))
        uids = set(pdb_struct.getData('uids'))
        if None in uids:
            uids.remove(None)
        uid2data = self.get_up_data(uids)
        uid2struct = pdb_mapping.extract_secondary_structures(pdb_id, uid2data)

        # Protein interactions.
        for u1c1, u2c2 in up_interactions:
            u1, c1 = u1c1
            u2, c2 = u2c2
            p1 = uid2data[u1]
            p2 = uid2data[u2]
            p1_struct = uid2struct[u1]
            p2_struct = uid2struct[u2]
            if p1['length'] < MIN_PROTEIN_LEN or p2['length'] < MIN_PROTEIN_LEN:
                continue
            pairs = list(up_interactions[(u1c1, u2c2)])
            yield {'pdb_id': pdb_id, 'p1_uni_id': p1['accession'],
                   'p2_uni_id': p2['accession'], 'p1_len': p1['length'],
                   'p2_len': p2['length'], 'p1_seq': p1['sequence'],
                   'p2_seq': p2['sequence'], 'p1_struct': p1_struct,
                   'p2_struct': p2_struct, 'organism': p2['organism'],
                   'pmid': pmid, 'interaction_type': 'protein-protein',
                   'p1_pfam': p1['pfam'], 'p2_pfam': p2['pfam'],
                   'p1_pdb_chain': c1, 'p2_pdb_chain': c2,
                   'interacting_residues': pairs}

    def pl_from_pdb_id(self, pdb_id):

        pdb_struct = parsePDB(pdb_id)

        reference = parsePDBHeader(pdb_id, 'reference')
        pmid = int(reference['pmid']) if 'pmid' in reference else 0

        pdb_mapping.map_atom_group(pdb_struct)

        up_interactions = self.map_interactions(pdb_id,
                self.get_interactions_list(pdb_struct))
        uids = set(pdb_struct.getData('uids'))
        if None in uids:
            uids.remove(None)
        uid2data = self.get_up_data(uids)
        uid2struct = pdb_mapping.extract_secondary_structures(pdb_id, uid2data)

        # Ligands interactions.
        chemicals = {}
        for c in parsePDBHeader(pdb_id, 'chemicals'):
            if not c.resname in chemicals:
                chemicals[c.resname] = c
        for chemical in chemicals.itervalues():
            ligand_interactions = self.get_ligand_interactions(pdb_id,
                                                          chemical.resname,
                                                          pdb_struct)
            for uid in ligand_interactions:
                if not uid in uid2data:
                    uid2data[uid] = self.up_parser.get_uniprot_entry(uid)
                p = uid2data[uid]
                p_struct = uid2struct[uid]
                yield {'ligand_name': chemical.name,
                        'ligand_formula': chemical.formula,
                        'pdb_id': pdb_id, 'p_uni_id': p['accession'],
                        'p_len': p['length'],
                        'p_seq': p['sequence'],
                        'p_struct': p_struct,
                        'p_pfam': p['pfam'],
                        'organism': p['organism'],
                        'pmid': pmid, 'interaction_type': 'protein-ligand',
                        'interacting_residues': list(ligand_interactions[uid])}

    def ptm_from_pdb_id(self, pdb_id):

        pdb_struct = parsePDB(pdb_id)

        reference = parsePDBHeader(pdb_id, 'reference')
        pmid = int(reference['pmid']) if 'pmid' in reference else 0

        pdb_mapping.map_atom_group(pdb_struct)

        up_interactions = self.map_interactions(pdb_id,
                self.get_interactions_list(pdb_struct))
        uids = set(pdb_struct.getData('uids'))
        if None in uids:
            uids.remove(None)
        uid2data = self.get_up_data(uids)
        uid2struct = pdb_mapping.extract_secondary_structures(pdb_id, uid2data)

        # PTMs.
        for u, p in uid2data.iteritems():
            ptm_residues = {}
            for ptm in p['ptm_features']:
                if not ptm[0] in ptm_residues:
                    ptm_residues[ptm[0]] = []
                ptm_residues[ptm[0]].extend(range(int(ptm[2][0]),
                                                  int(ptm[2][1])+1))
            p_struct = uid2struct[u]
            for ptm in ptm_residues:
                yield {'ptm_type': ptm,
                       'pdb_id': pdb_id, 'p_uni_id': p['accession'],
                       'p_len': p['length'],
                       'p_seq': p['sequence'],
                       'p_struct': p_struct,
                       'p_pfam': p['pfam'],
                       'organism': p['organism'],
                       'pmid': pmid, 'interaction_type': 'ptm',
                       'affected_residues': ptm_residues[ptm]}


    def add_pp_entry(self, cursor, e):
        '''Adds entry into `protein-protein` interactions table.'''
        try:
            t = (e['pdb_id'], e['p1_uni_id'], e['p2_uni_id'], e['organism'],
                 e['pmid'], e['p1_len'], e['p1_seq'],
                 e['p1_struct'], e['p1_pfam'], e['p2_len'], e['p2_seq'],
                 e['p2_struct'], e['p2_pfam'], json.dumps(e['interacting_residues']),
                 e['p1_pdb_chain'], e['p2_pdb_chain'],
                 '', '')
            cursor.execute('''INSERT INTO `protein-protein` VALUES
                            (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', t)
        except sqlite3.IntegrityError:
            print("Entry {0} {1} {2} already exists.".format(e['pdb_id'],e['p1_uni_id'],e['p2_uni_id']))

    def add_pl_entry(self, cursor, e):
        '''Adds entry into `protein-ligand` interactions table.'''
        try:
            t = (e['pdb_id'], e['p_uni_id'], e['ligand_name'],
                 e['ligand_formula'],
                 e['organism'], e['pmid'], e['p_len'], e['p_seq'],
                 e['p_struct'], e['p_pfam'],
                 json.dumps(e['interacting_residues']), '', '')
            cursor.execute('''INSERT INTO `protein-ligand` VALUES
                            (?,?,?,?,?,?,?,?,?,?,?,?,?)''', t)
        except sqlite3.IntegrityError:
            print("Entry {0} {1} {2} already exists.".format(e['pdb_id'],e['p_uni_id'],e['ligand_name']))

    def add_ptm_entry(self, cursor, e):
        '''Adds entry into `ptm` table.'''
        try:
            t = (e['pdb_id'], e['p_uni_id'], e['organism'], e['ptm_type'],
                 e['pmid'], e['p_len'], e['p_seq'], e['p_struct'], e['p_pfam'],
                 json.dumps(e['affected_residues']), '', '')
            cursor.execute('''INSERT INTO `ptm` VALUES
                            (?,?,?,?,?,?,?,?,?,?,?,?)''', t)
        except sqlite3.IntegrityError:
            print("Entry {0} {1} {2} already exists.".format(e['pdb_id'],e['p_uni_id'],e['ptm_type']))

    def create_protein_table(self, cursor):
        cursor.execute('''CREATE TABLE `protein-protein`
                 (pdb_id text, p1_uni_id text, p2_uni_id text, organism text,
                  pmid int, p1_len int,
                  p1_seq text, p1_struct text, p1_pfam text, p2_len int,
                  p2_seq text,
                  p2_struct text, p2_pfam text, interacting_residues text,
                  p1_pdb_chain text, p2_pdb_chain text,
                  cath_id text default '',
                  scop_id text default '',
                  PRIMARY KEY(pdb_id, p1_uni_id, p2_uni_id, p1_pdb_chain, p2_pdb_chain))''')

    def create_ligand_table(self, cursor):
        cursor.execute('''CREATE TABLE `protein-ligand`
                 (pdb_id text, p_uni_id text, ligand_name text,
                  ligand_formula text,
                  organism text, pmid int, p_len int,
                  p_seq text, p_struct text, p_pfam text,
                  interacting_residues text,
                  cath_id text default '', scop_id text default '',
                  PRIMARY KEY(pdb_id, p_uni_id, ligand_name))''')

    def create_ptm_table(self, cursor):
        cursor.execute('''CREATE TABLE `ptm`
                 (pdb_id text, p_uni_id text, organism text, ptm_name text,
                  pmid int, p_len int, p_seq text, p_struct text, p_pfam text,
                  affected_residues text, cath_id text default '',
                  scop_id text default '',
                  PRIMARY KEY(pdb_id, p_uni_id, ptm_name))''')

    def close(self):
        pass
