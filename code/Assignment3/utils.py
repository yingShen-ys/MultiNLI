


str = "(ROOT (S (NP (PRP$ Your) (NNS contributions)) (VP (VBD were) (PP (IN of) (NP (NP (DT no) (NN help)) (PP (IN with) (NP (NP (PRP$ our) (NNS students) (POS ')) (NN education)))))) (. .)))"
def generate_pos_tag_from_parse_tree(parse_tree):
    pos_tags = []
    words = []
    items = parse_tree.split(' ')
    for idx, item in enumerate(items):
        if item.endswith(')'):
            pos_tags.append(items[idx-1].replace('(', ''))
            words.append(item.replace(')', ''))
    print(pos_tags)
    print(words)
    return pos_tags

generate_pos_tag_from_parse_tree(str)
