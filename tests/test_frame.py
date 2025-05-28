import pandas as pd
import numpy as np
import pytest
import pyturbo as pt
from pyturbo.core.frame import TurboFrame, HAS_GPU, TurboGroupBy # For inspecting internals, HAS_GPU and TurboGroupBy
from pyturbo.config import get_config, set_num_threads # For checking config
import pandas.testing as tm

# Conditional import of cuDF
if HAS_GPU:
    import cudf
else:
    cudf = None

# --- Fixtures for Sample Data ---

@pytest.fixture
def customers_pdf_small():
    """Small sample customers Pandas DataFrame (for direct/CPU tests)."""
    return pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5, 6],
        'age': [25, 34, 28, 42, 50, 25], # Added a duplicate age for sort testing
        'region': ['North', 'South', 'East', 'West', 'North', 'South'],
        'income': [50000, 62000, 55000, 75000, 48000, 51000]
    })

@pytest.fixture
def transactions_pdf_small():
    """Small sample transactions Pandas DataFrame."""
    return pd.DataFrame({
        'transaction_id': range(12), # Increased size slightly
        'customer_id': [1, 2, 1, 3, 2, 4, 1, 5, 3, 4, 6, 6],
        'amount': [100, 150, 50, 200, 120, 300, 70, 90, 250, 180, 50, 60],
        'category': ['Food', 'Electronics', 'Food', 'Books', 'Electronics', 'Books', 'Food', 'Clothing', 'Books', 'Electronics', 'Food', 'Other']
    })

@pytest.fixture
def customers_pdf_large():
    """Larger customers Pandas DataFrame (to trigger Dask strategies)."""
    # Approx 5000 rows, should trigger threaded Dask (direct_threshold=1000, thread_threshold=100k elements)
    # 5000 rows * 4 cols = 20000 elements.
    num_rows = 5000
    return pd.DataFrame({
        'customer_id': range(num_rows),
        'age': np.random.randint(18, 90, num_rows),
        'region': np.random.choice(['North', 'South', 'East', 'West'], num_rows),
        'income': np.random.normal(60000, 20000, num_rows)
    })

@pytest.fixture
def transactions_pdf_large():
    """Larger transactions Pandas DataFrame."""
    num_rows = 5000  # Must match customers_pdf_large for meaningful merges
    num_transactions = num_rows * 3 # Approx 15k transactions
    return pd.DataFrame({
        'transaction_id': range(num_transactions),
        'customer_id': np.random.choice(num_rows, num_transactions), # Use customer_ids from 0 to num_rows-1
        'amount': np.random.normal(100, 50, num_transactions),
        'category': np.random.choice(['Food', 'Electronics', 'Clothing', 'Books', 'Other'], num_transactions)
    })


# --- Helper function for DataFrame comparison ---
def assert_df_equal(df1, df2, sort_by_columns=True, check_like=False, **kwargs):
    """
    Asserts two DataFrames are equal. Handles cuDF conversion to Pandas.
    kwargs are passed to tm.assert_frame_equal.
    If sort_by_columns is True, sorts columns and then rows by all columns.
    If check_like is True, it will sort columns and reset index, useful when row order is not guaranteed.
    """
    df1_processed = df1.copy()
    df2_processed = df2.copy()

    if HAS_GPU and cudf:
        if isinstance(df1_processed, cudf.DataFrame):
            df1_processed = df1_processed.to_pandas()
        if isinstance(df2_processed, cudf.DataFrame):
            df2_processed = df2_processed.to_pandas()
            
    if sort_by_columns and not df1_processed.empty and not df2_processed.empty:
        common_cols = sorted(list(set(df1_processed.columns) & set(df2_processed.columns)))
        if not common_cols:
             tm.assert_frame_equal(df1_processed.reset_index(drop=True), df2_processed.reset_index(drop=True), **kwargs)
             return
        df1_processed = df1_processed[common_cols].sort_values(by=common_cols).reset_index(drop=True)
        df2_processed = df2_processed[common_cols].sort_values(by=common_cols).reset_index(drop=True)
    elif check_like: # Sorts columns, resets index, but doesn't sort values
        df1_processed = df1_processed.sort_index(axis=1).reset_index(drop=True)
        df2_processed = df2_processed.sort_index(axis=1).reset_index(drop=True)
    else: 
        df1_processed = df1_processed.reset_index(drop=True)
        df2_processed = df2_processed.reset_index(drop=True)
        
    if 'check_dtype' not in kwargs:
        kwargs['check_dtype'] = False # Often types differ slightly (int32 vs int64)
        
    tm.assert_frame_equal(df1_processed, df2_processed, **kwargs)


# --- Lazy Evaluation Tests ---
# (Using small fixtures as these test laziness, not performance or large data paths)

def test_merge_is_lazy(customers_pdf_small, transactions_pdf_small):
    tf_customers = pt.TurboFrame(customers_pdf_small)
    tf_transactions = pt.TurboFrame(transactions_pdf_small)
    result_tf = tf_customers.merge(tf_transactions, on='customer_id')
    assert isinstance(result_tf, TurboFrame)
    assert len(result_tf._ops_pipeline) > 0 and result_tf._ops_pipeline[-1][0] == 'merge'
    assert result_tf._computed_data is None
    assert isinstance(result_tf.compute(), pd.DataFrame)

def test_groupby_agg_is_lazy(customers_pdf_small): # Updated for TurboGroupBy
    tf_customers = pt.TurboFrame(customers_pdf_small)
    grouped_obj = tf_customers.groupby('region')
    assert isinstance(grouped_obj, TurboGroupBy) # groupby() returns TurboGroupBy
    
    result_tf = grouped_obj.agg({'age': 'mean'}) # agg() returns TurboFrame
    assert isinstance(result_tf, TurboFrame)
    assert len(result_tf._ops_pipeline) > 0 and result_tf._ops_pipeline[-1][0] == 'groupby_agg'
    assert result_tf._computed_data is None
    assert isinstance(result_tf.compute(), pd.DataFrame)

def test_sort_values_is_lazy(customers_pdf_small):
    tf_customers = pt.TurboFrame(customers_pdf_small)
    result_tf = tf_customers.sort_values(by='age')
    assert isinstance(result_tf, TurboFrame)
    assert len(result_tf._ops_pipeline) > 0 and result_tf._ops_pipeline[-1][0] == 'sort_values'
    assert result_tf._computed_data is None
    assert isinstance(result_tf.compute(), pd.DataFrame)

def test_getitem_is_lazy(customers_pdf_small):
    tf_customers = pt.TurboFrame(customers_pdf_small)
    result_tf = tf_customers['age'] 
    assert isinstance(result_tf, TurboFrame)
    assert len(result_tf._ops_pipeline) > 0 and result_tf._ops_pipeline[-1][0] == 'select'
    assert result_tf._computed_data is None
    assert isinstance(result_tf.compute(), pd.DataFrame) # getitem now returns DataFrame

def test_chaining_is_lazy(customers_pdf_small, transactions_pdf_small):
    tf_customers = pt.TurboFrame(customers_pdf_small)
    tf_transactions = pt.TurboFrame(transactions_pdf_small)
    
    result_tf = tf_customers.merge(tf_transactions, on='customer_id') \
                              .groupby('category') \
                              .agg({'amount': 'sum'}) \
                              .sort_values(by='amount', ascending=False)
    assert isinstance(result_tf, TurboFrame)
    assert len(result_tf._ops_pipeline) == 3 # merge, groupby_agg, sort_values
    assert result_tf._ops_pipeline[0][0] == 'merge'
    assert result_tf._ops_pipeline[1][0] == 'groupby_agg'
    assert result_tf._ops_pipeline[2][0] == 'sort_values'
    assert result_tf._computed_data is None
    assert isinstance(result_tf.compute(), pd.DataFrame)


# --- Correctness Tests (CPU - Direct Path with Small Data) ---

def test_merge_correctness_direct_cpu(customers_pdf_small, transactions_pdf_small):
    tf_cust = pt.TurboFrame(customers_pdf_small)
    tf_trans = pt.TurboFrame(transactions_pdf_small)
    
    pt_result = tf_cust.merge(tf_trans, on='customer_id', how='left').compute()
    pd_result = customers_pdf_small.merge(transactions_pdf_small, on='customer_id', how='left')
    assert_df_equal(pt_result, pd_result, sort_by_columns=['customer_id', 'transaction_id'])

def test_groupby_agg_correctness_direct_cpu(customers_pdf_small): # Unskipped and updated
    tf_cust = pt.TurboFrame(customers_pdf_small)
    agg_dict = {'age': 'mean', 'income': ['sum', 'std']}
    
    pt_result = tf_cust.groupby('region').agg(agg_dict).compute()
    pd_result = customers_pdf_small.groupby('region').agg(agg_dict)
    # Pandas MultiIndex columns: rename for easier comparison
    pd_result.columns = ['_'.join(col).strip() for col in pd_result.columns.values]
    pd_result = pd_result.reset_index()
    
    assert_df_equal(pt_result, pd_result, sort_by_columns=['region'])

def test_sort_values_correctness_direct_cpu(customers_pdf_small):
    tf_cust = pt.TurboFrame(customers_pdf_small)
    pt_result = tf_cust.sort_values(by=['region', 'age'], ascending=[True, False]).compute()
    pd_result = customers_pdf_small.sort_values(by=['region', 'age'], ascending=[True, False])
    assert_df_equal(pt_result, pd_result, sort_by_columns=False)

def test_getitem_correctness_direct_cpu(customers_pdf_small):
    tf_cust = pt.TurboFrame(customers_pdf_small)
    # Single column
    pt_result_single = tf_cust['age'].compute()
    pd_result_single = customers_pdf_small[['age']]
    assert_df_equal(pt_result_single, pd_result_single, sort_by_columns=False)
    # Multiple columns
    cols = ['age', 'region']
    pt_result_multi = tf_cust[cols].compute()
    pd_result_multi = customers_pdf_small[cols]
    assert_df_equal(pt_result_multi, pd_result_multi, sort_by_columns=False)

def test_chaining_correctness_direct_cpu(customers_pdf_small, transactions_pdf_small):
    tf_cust = pt.TurboFrame(customers_pdf_small)
    tf_trans = pt.TurboFrame(transactions_pdf_small)
    agg_spec = {'amount': ['sum', 'mean'], 'age': 'max'}

    pt_result = tf_cust.merge(tf_trans, on='customer_id', how='left') \
                       .groupby('category') \
                       .agg(agg_spec) \
                       .sort_values(by=('amount_sum'), ascending=False) \
                       [['amount_sum', 'amount_mean', 'age_max']] \
                       .compute()

    pd_merged = customers_pdf_small.merge(transactions_pdf_small, on='customer_id', how='left')
    pd_grouped = pd_merged.groupby('category').agg(agg_spec)
    pd_grouped.columns = ['_'.join(col).strip() for col in pd_grouped.columns.values]
    pd_result = pd_grouped.sort_values(by='amount_sum', ascending=False) \
                          [['amount_sum', 'amount_mean', 'age_max']]
                          
    assert_df_equal(pt_result, pd_result, sort_by_columns=False) # Order from sort_values


# --- Correctness Tests (CPU - Dask Path with Large Data) ---
# These use larger fixtures to encourage ComputeEngine to use Dask strategies

def test_merge_correctness_dask_cpu(customers_pdf_large, transactions_pdf_large):
    tf_cust = pt.TurboFrame(customers_pdf_large)
    tf_trans = pt.TurboFrame(transactions_pdf_large)
    
    pt_result = tf_cust.merge(tf_trans, on='customer_id', how='left').compute()
    pd_result = customers_pdf_large.merge(transactions_pdf_large, on='customer_id', how='left')
    assert_df_equal(pt_result, pd_result, sort_by_columns=['customer_id', 'transaction_id'])

def test_groupby_agg_correctness_dask_cpu(customers_pdf_large):
    tf_cust = pt.TurboFrame(customers_pdf_large)
    agg_dict = {'age': 'mean', 'income': ['sum', 'std']}
    
    pt_result = tf_cust.groupby('region').agg(agg_dict).compute()
    pd_result = customers_pdf_large.groupby('region').agg(agg_dict)
    pd_result.columns = ['_'.join(col).strip() for col in pd_result.columns.values]
    pd_result = pd_result.reset_index()
    assert_df_equal(pt_result, pd_result, sort_by_columns=['region'])

def test_sort_values_correctness_dask_cpu(customers_pdf_large):
    tf_cust = pt.TurboFrame(customers_pdf_large)
    pt_result = tf_cust.sort_values(by=['region', 'age'], ascending=[True, False]).compute()
    pd_result = customers_pdf_large.sort_values(by=['region', 'age'], ascending=[True, False])
    assert_df_equal(pt_result, pd_result, sort_by_columns=False)

def test_getitem_correctness_dask_cpu(customers_pdf_large):
    tf_cust = pt.TurboFrame(customers_pdf_large)
    cols = ['age', 'region']
    pt_result_multi = tf_cust[cols].compute()
    pd_result_multi = customers_pdf_large[cols]
    assert_df_equal(pt_result_multi, pd_result_multi, sort_by_columns=False)

def test_chaining_correctness_dask_cpu(customers_pdf_large, transactions_pdf_large):
    tf_cust = pt.TurboFrame(customers_pdf_large)
    tf_trans = pt.TurboFrame(transactions_pdf_large)
    agg_spec = {'amount': ['sum', 'mean'], 'age': 'max'}

    pt_result = tf_cust.merge(tf_trans, on='customer_id', how='left') \
                       .groupby('category') \
                       .agg(agg_spec) \
                       .sort_values(by=('amount_sum'), ascending=False) \
                       [['amount_sum', 'amount_mean', 'age_max']] \
                       .compute()

    pd_merged = customers_pdf_large.merge(transactions_pdf_large, on='customer_id', how='left')
    pd_grouped = pd_merged.groupby('category').agg(agg_spec)
    pd_grouped.columns = ['_'.join(col).strip() for col in pd_grouped.columns.values]
    pd_result = pd_grouped.sort_values(by='amount_sum', ascending=False) \
                          [['amount_sum', 'amount_mean', 'age_max']]
    assert_df_equal(pt_result, pd_result, sort_by_columns=False)


# --- GPU Tests ---
skip_if_no_gpu = pytest.mark.skipif(not HAS_GPU, reason="GPU (cuDF) not available or not installed")

@skip_if_no_gpu
def test_data_to_gpu_on_init_with_context(customers_pdf_small):
    with pt.use_gpu():
        tf_gpu = pt.TurboFrame(customers_pdf_small)
        assert isinstance(tf_gpu._initial_data, cudf.DataFrame)
        assert tf_gpu._device == 'gpu'

@skip_if_no_gpu
def test_data_to_gpu_manual(customers_pdf_small):
    tf_cpu = pt.TurboFrame(customers_pdf_small)
    assert isinstance(tf_cpu._initial_data, pd.DataFrame)
    
    tf_gpu = tf_cpu.to_gpu()
    assert tf_gpu._device == 'gpu'
    # If no ops pending, _initial_data should be transferred
    if not tf_gpu._ops_pipeline:
        assert isinstance(tf_gpu._initial_data, cudf.DataFrame)
    # Check compute also yields cudf
    assert isinstance(tf_gpu.compute(), cudf.DataFrame)


@skip_if_no_gpu
def test_merge_correctness_gpu(customers_pdf_small, transactions_pdf_small):
    with pt.use_gpu():
        tf_cust_gpu = pt.TurboFrame(customers_pdf_small)
        tf_trans_gpu = pt.TurboFrame(transactions_pdf_small)
        
        pt_result_gpu = tf_cust_gpu.merge(tf_trans_gpu, on='customer_id', how='left').compute()
        
    pd_result = customers_pdf_small.merge(transactions_pdf_small, on='customer_id', how='left')
    # assert_df_equal will convert pt_result_gpu (cuDF) to pandas for comparison
    assert_df_equal(pt_result_gpu, pd_result, sort_by_columns=['customer_id', 'transaction_id'])


@skip_if_no_gpu
def test_groupby_agg_correctness_gpu(customers_pdf_small):
    agg_dict = {'age': 'mean', 'income': ['sum', 'std']}
    with pt.use_gpu():
        tf_cust_gpu = pt.TurboFrame(customers_pdf_small)
        pt_result_gpu = tf_cust_gpu.groupby('region').agg(agg_dict).compute()

    pd_result = customers_pdf_small.groupby('region').agg(agg_dict)
    pd_result.columns = ['_'.join(col).strip() for col in pd_result.columns.values]
    pd_result = pd_result.reset_index()
    assert_df_equal(pt_result_gpu, pd_result, sort_by_columns=['region'])


@skip_if_no_gpu
def test_sort_values_correctness_gpu(customers_pdf_small):
    with pt.use_gpu():
        tf_cust_gpu = pt.TurboFrame(customers_pdf_small)
        pt_result_gpu = tf_cust_gpu.sort_values(by=['region', 'age'], ascending=[True, False]).compute()
        
    pd_result = customers_pdf_small.sort_values(by=['region', 'age'], ascending=[True, False])
    assert_df_equal(pt_result_gpu, pd_result, sort_by_columns=False)


@skip_if_no_gpu
def test_getitem_correctness_gpu(customers_pdf_small):
    cols = ['age', 'region']
    with pt.use_gpu():
        tf_cust_gpu = pt.TurboFrame(customers_pdf_small)
        pt_result_gpu = tf_cust_gpu[cols].compute()

    pd_result = customers_pdf_small[cols]
    assert_df_equal(pt_result_gpu, pd_result, sort_by_columns=False)


@skip_if_no_gpu
def test_chaining_correctness_gpu(customers_pdf_small, transactions_pdf_small):
    agg_spec = {'amount': ['sum', 'mean'], 'age': 'max'}
    with pt.use_gpu():
        tf_cust_gpu = pt.TurboFrame(customers_pdf_small)
        tf_trans_gpu = pt.TurboFrame(transactions_pdf_small)
        pt_result_gpu = tf_cust_gpu.merge(tf_trans_gpu, on='customer_id', how='left') \
                                   .groupby('category') \
                                   .agg(agg_spec) \
                                   .sort_values(by=('amount_sum'), ascending=False) \
                                   [['amount_sum', 'amount_mean', 'age_max']] \
                                   .compute()

    pd_merged = customers_pdf_small.merge(transactions_pdf_small, on='customer_id', how='left')
    pd_grouped = pd_merged.groupby('category').agg(agg_spec)
    pd_grouped.columns = ['_'.join(col).strip() for col in pd_grouped.columns.values]
    pd_result = pd_grouped.sort_values(by='amount_sum', ascending=False) \
                          [['amount_sum', 'amount_mean', 'age_max']]
    assert_df_equal(pt_result_gpu, pd_result, sort_by_columns=False)

```
