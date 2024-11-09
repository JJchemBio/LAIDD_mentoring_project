from multiprocessing import Pool
import itertools

def multiproc_task_on_list(task, list_input, n_jobs):
    """
        task: function that applies to each element of the list
        list_input: list of elements
        n_jobs: number of subprocesses to be used
    """
    proc_pool = Pool(n_jobs)
    list_output = proc_pool.map(task, list_input)
    proc_pool.close()
    return list_output

def pairwise_tupled_ops(task, list1, list2, n_jobs):
    """
        task: function that applies to each pair of elements. The input is a single tuple.
        list1: first elem of a tuple
        list2: second elem of a tuple
        n_jobs: number of subprocesses to be used
        > returns:
            - re_matrix: list of list, where row is list1 elem, col is list2 elem.
    """
    rs, cs = len(list1), len(list2)  # row size, column size
    tup_list = list(itertools.product(list1, list2))
    flat_paired = multiproc_task_on_list(task, tup_list, n_jobs)
    re_matrix = []
    for i in range(rs):
        row_start = cs*i
        re_matrix.append(flat_paired[row_start:(row_start+cs)])
    return re_matrix
    