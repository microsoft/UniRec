using System;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace inference_csharp
{
  
    /// <summary>
    /// data for one user's history
    /// </summary>
    public class History_Node
    {
        public long user_id { get; set; }
        public long[] item_seq { get; set; } = null!;
        public long[] time_seq { get; set; } = null!;
        public bool has_time { get; set; }
    }

    /// <summary>
    /// data for one item's features
    /// </summary>
    public class Feature_Node
    {
        public long item_id { get; set; }
        public long[] item_feats { get; set; } = null!;
        public Feature_Node()
        {
            item_id = 0;
            item_feats = new long[GlobalVar.n_features];
            for(int i = 0; i < GlobalVar.n_features; i++)
            {
                item_feats[i] = 0;
            }
        }
    }

    /// <summary>
    /// one test case
    /// </summary>
    public class Test_Node
    {
        public long user_id { get; set; }
        public long item_id { get; set; }
    }

    /// <summary>
    /// collect for all user history and test case
    /// </summary>
    public class Data
    {
        public History_Node[] user_history { get; set; } = null!;
        public Feature_Node[] item_features { get; set; } = null!;
        public List<Test_Node> test_set { get; set; } = null!;
        public Data()
        {
            user_history = new History_Node[GlobalVar.n_users];
            test_set = new List<Test_Node>();
        }
    }

    /// <summary>
    /// input for one batch, default value is all zero
    /// </summary>
    public class Batch_Input
    {
        // 'user_id' 'item_id' 'item_features' 'item_seq' 'item_seq_features'
        public Tensor<long> user_id { get; set; } = null!;
        public Tensor<long> item_id { get; set; } = null!;
        public Tensor<long> item_features { get; set; } = null!;
        public Tensor<long> item_seq { get; set; } = null!;
        public Tensor<long> item_seq_len { get; set; } = null!;
        public Tensor<long> item_seq_features { get; set; } = null!;
        public Tensor<long> time_seq { get; set; } = null!;
        public Batch_Input(int batch_size)
        {
            user_id = new DenseTensor<long>(new[] { batch_size });
            item_id = new DenseTensor<long>(new[] { batch_size });
            item_seq = new DenseTensor<long>(new[] { batch_size, GlobalVar.max_seq_len });
            item_seq_len = new DenseTensor<long>(new[] { batch_size });
            time_seq = new DenseTensor<long>(new[] { batch_size, GlobalVar.max_seq_len });
            for (int i = 0; i < batch_size; i++)
            {
                for(int j = 0; j < GlobalVar.max_seq_len; j++)
                {
                    item_seq[i, j] = 0;
                }
            }
            for (int i = 0; i < batch_size; i++)
            {
                item_seq_len[i] = 0;
            }
            for (int i = 0; i < batch_size; i++)
            {
                for(int j = 0; j < GlobalVar.max_seq_len; j++)
                {
                    time_seq[i, j] = 0;
                }
            }
        }
    }
}
