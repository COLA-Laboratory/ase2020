# include, exclude: dict    pipeline: dict
def in_ex_cludeHandler(include, exclude, pipeline):
    if include is not None:
        for key, val in include.items():
            delist = []
            for k, v in pipeline[key].items():
                if k not in val:
                    delist.append(k)
            for item in delist:
                del pipeline[key][item]

    if exclude is not None:
        for key, value in exclude.items():
            delist = []
            for k, v in pipeline[key].items():
                if k in value:
                    delist.append(k)
            for item in delist:
                del pipeline[key][item]
    return pipeline

