#! /usr/bin/env python
###################
# Read data from a Uniprot XML dump
###################

import urllib
import sys, re, textwrap, operator, os
import xml.etree.ElementTree as ET

ns = '{http://uniprot.org/uniprot}'
data_folder = os.path.dirname(os.path.abspath(__file__)) + '/up_data'

class UPParser(object):

    def __init__(self):
        self.cache = {} 

    def get_name(self, protein):
        '''Get the name of the protein'''
        try:
            name = protein.find(ns+'protein').find(ns+'recommendedName').findtext(ns+'fullName')
        except:
            name = protein.find(ns+'protein').find(ns+'submittedName').findtext(ns+'fullName')
        return name

    def get_gene_name(self, protein):
        '''Get the gene name'''
        try:
            gene_name = protein.find(ns+'gene').findtext(ns+'name')
        except:
            # Have to invent short names for proteins that don't have one
            try:
                fullname = protein.find(ns+'protein').find(ns+'recommendedName').findtext(ns+'fullName')
            except:
                fullname = protein.find(ns+'protein').find(ns+'submittedName').findtext(ns+'fullName')
            gene_name = fullname[0:13]
        return gene_name
        
    def get_accession(self, protein):
        '''Get accession code for this protein'''
        accession = protein.find(ns+'accession')
        return accession.text

    def get_sequence(self, protein):
        '''Retrieve the protein sequence'''
        sequence = protein.findtext(ns+'sequence')
        return re.sub("\s+", "", sequence)

    def get_pfams(self, protein):
        '''Retrieve the Pfam families'''
        references = protein.findall(ns+'dbReference')
        pfams = []
        for ref in references:
            if ref.get('type') == 'Pfam':
                pfams.append(ref.get('id'))
        return pfams

    def get_structure(self, protein):
        '''Return structure with start and stop positions'''
        s_list = []
        features = protein.findall(ns+'feature')
        for feature in features:
            type = feature.get('type')
            if type in ['turn', 'strand', 'helix']:
                s_start = feature.find(ns+'location').find(ns+'begin').get('position')
                s_stop = feature.find(ns+'location').find(ns+'end').get('position')
                s_list.append((type, int(s_start), int(s_stop)))
        return s_list

    def get_ptm_features(self, protein):
        '''Return a list of various amino acid modifications'''
        ptm_list = []
        for feature in protein.findall(ns+'feature'):
            type = feature.get('type')
            ptm_types = ['non-standard residue', 'modified residue', 'lipidation',
                         'glycosylation', 'disulfide bond', 'cross-link']
            if type in ptm_types:
                loc = feature.find(ns+'location')
                if loc.find(ns+'position') is not None:
                    pos = loc.find(ns+'position').get('position')
                    range = (pos, pos)
                else:
                    begin = loc.find(ns+'begin').get('position')
                    end = loc.find(ns+'end').get('position')
                    range = (begin, end)
                ptm_list.append((type, feature.get('description'), range))
        return ptm_list   

    def get_interactions(self, protein, prot_id):
        '''Returns a list of proteins interacting with the given one'''
        s = set()
        for comment in protein.findall(ns+'comment'):
            if comment.get('type') == 'interaction':
                for interact in comment.findall(ns+'interactant'):
                    id = interact.get('id')
                    if id:
                        s.add(id)
                    else:
                        s.add(prot_id)
        return list(s)
     
    def get_domain(self, protein):
        '''Return domains with start and stop positions'''
        d_list = []
        features = protein.findall(ns+'feature')
        for feature in features:
            if feature.get('type') == 'domain':
                d_start = feature.find(ns+'location').find(ns+'begin').get('position')
                d_stop = feature.find(ns+'location').find(ns+'end').get('position')
                desc = feature.get('description')
                d_list.append((desc, d_start, d_stop))
        return d_list

    def get_organism(self, protein):
        '''Return scientific organism name'''
        organism = protein.find(ns+'organism')
        return organism.find(ns+'name').text

    def structure_to_string(self, struct_tuples, seq_len):
        '''Converts protein structure from list of tuples format to string'''
        struct_tuples.sort(key=operator.itemgetter(1))
        struct_list = []
        for t in struct_tuples:
            if len(struct_list) < t[1]-1:
                struct_list.extend(['-' for x in range(len(struct_list), t[1]-1)])
            s = t[0][0].upper()
            struct_list.extend([s for x in range(t[1],t[2]+1)])
        if len(struct_list) < seq_len:
            struct_list.extend(['-' for x in range(len(struct_list), seq_len)])
        return "".join(struct_list)

    def get_uniprot_entry(self, id):
        '''Accepts UNIPROT id and returns dictionary of protein properties'''
        if id in self.cache:
            return self.cache[id]
        else:
            file_name = "{0}/{1}.xml".format(data_folder, id)
            urllib.urlretrieve("http://www.uniprot.org/uniprot/{0}.xml".format(id), file_name)
            try:
                tree = ET.parse(file_name)
            except:
                print "Error while parsing {0}".format(id)
                return None
            tree_root = tree.getroot()
            protein = tree_root.findall(ns+'entry')[0]
            p_dict = {}
            p_dict["name"] = self.get_name(protein)
            p_dict["gene_name"] = self.get_gene_name(protein)
            p_dict["organism"] = self.get_organism(protein)
            p_dict["accession"] = self.get_accession(protein)
            p_dict["sequence"] = self.get_sequence(protein)
            p_dict["domain"] = self.get_domain(protein)
            p_dict["interactions"] = self.get_interactions(protein, id)
            p_dict["ptm_features"] = self.get_ptm_features(protein)
            p_dict["length"] = len(p_dict["sequence"])
            p_dict["structure"] = self.structure_to_string(
                                    self.get_structure(protein),
                                    p_dict["length"])
            pfam_list = self.get_pfams(protein)
            if pfam_list:
                p_dict["pfam"] = pfam_list[0]
            else:
                p_dict["pfam"] = ""
            self.cache[id] = p_dict
            return p_dict
