

def set_operator_type(rec_operator_type, t_set):
    t_set["rec_operator_type"] = rec_operator_type
    if rec_operator_type == "full_linear":
        t_set["full_linear"] = True
        t_set["skip_linear"] = False
        t_set["not_skip_linear_diagonal"] = False
    elif rec_operator_type == "diagonal":
        t_set["full_linear"] = False
        t_set["skip_linear"] = False
        t_set["not_skip_linear_diagonal"] = True
    elif rec_operator_type == "factor_UUt":
        t_set["full_linear"] = False
        t_set["skip_linear"] = False
        t_set["not_skip_linear_diagonal"] = False
    elif rec_operator_type == "factor_zeros":
        t_set["full_linear"] = False
        t_set["skip_linear"] = True
        t_set["not_skip_linear_diagonal"] = False
    else:
        print(f"Unknown rec_operator_type: {rec_operator_type}")