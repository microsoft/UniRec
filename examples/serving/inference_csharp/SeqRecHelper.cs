using System;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Linq;
using System.IO;


namespace inference_csharp
{

    class Utilities
    {
        /// <summary>
        /// load user history and test set
        /// </summary>
        public static Data LoadData(string history_file, string feature_file, string test_file)
        {
            
            Data data = new Data();
            using (StreamReader sr = new StreamReader(history_file))
            {
                var line = sr.ReadLine();
                while (true)
                {
                    line = sr.ReadLine();
                    if (line == null)
                        break;

                    History_Node his_node = new History_Node();
                    string[] list = line.Trim().Split("\t");
                    his_node.user_id = Convert.ToInt64(list[0].Trim());
                    his_node.item_seq = list[1].Trim().Split(",").Select(x => Convert.ToInt64(x.Trim())).ToArray();
                    if (list.Length == 3)
                    {   
                        his_node.has_time = true;
                        his_node.time_seq = list[2].Trim().Split(",").Select(x => Convert.ToInt64(x.Trim())).ToArray();
                    }
                    else
                        {his_node.has_time = false;}
                    data.user_history[his_node.user_id] = his_node;
                }
            }

            if (GlobalVar.has_feature)
            {
                data.item_features = new Feature_Node[GlobalVar.n_items];
                data.item_features[0] = new Feature_Node();
                using (StreamReader sr = new StreamReader(feature_file))
                {
                    var line = sr.ReadLine();
                    while (true)
                    {
                        line = sr.ReadLine();
                        if (line == null)
                            break;

                        Feature_Node fea_node = new Feature_Node();
                        string[] list = line.Trim().Split("\t");
                        fea_node.item_id = Convert.ToInt64(list[0].Trim());
                        fea_node.item_feats = list[1].Trim().Split(",").Select(x => Convert.ToInt64(x.Trim())).ToArray();
                        data.item_features[fea_node.item_id] = fea_node;
                    }
                }

            }

            using (StreamReader sr = new StreamReader(test_file))
            {
                var line = sr.ReadLine();
                while (true)
                {
                    line = sr.ReadLine();
                    if (line == null)
                        break;

                    Test_Node t_node = new Test_Node();

                    string[] list = line.Trim().Split("\t");
                    t_node.user_id = Convert.ToInt64(list[0].Trim());
                    t_node.item_id = Convert.ToInt64(list[1].Trim());

                    data.test_set.Add(t_node);
                }
            }

            return data;
        }


        /// <summary>
        /// collate for one batch data, including item_seq, item_seq_len, time_seq(if exists)
        /// default pad left with 0
        /// currently only support 'autoregressive' history_mask_mode
        /// l and r are the left and right index in the test set corresponding to the batch
        /// </summary>
        public static Batch_Input collate_batch(Data data, int l, int r)
        {
            Batch_Input batch_input = new Batch_Input(r-l);
            for (int i=l; i<r; i++)
            {
                batch_input.user_id[i-l] = data.test_set[i].user_id;
                batch_input.item_id[i-l] = data.test_set[i].item_id;
                History_Node history = data.user_history[data.test_set[i].user_id];
                long item_id = data.test_set[i].item_id;
                int history_len = history.item_seq.Length-1;
                for (; history_len>=0; history_len--)
                {
                    if (history.item_seq[history_len] == item_id)
                        {break;}
                }
                if (history_len == -1)
                    {history_len = history.item_seq.Length;}
                batch_input.item_seq_len[i-l] = Math.Min(history_len, GlobalVar.max_seq_len);
                for ( int j=history_len-1, k=GlobalVar.max_seq_len-1 ; j>=Math.Max(0, history_len-GlobalVar.max_seq_len); j--, k-- )
                {
                    batch_input.item_seq[i-l, k] = history.item_seq[j];
                    if (history.has_time)
                        {batch_input.time_seq[i-l, k] = history.time_seq[j];}
                }
            }
            if (GlobalVar.has_feature)
            {
                batch_input.item_features = new DenseTensor<long>(new[] { r-l, GlobalVar.n_features });
                for (int i=l; i<r; i++)
                {
                    for (int j=0; j<GlobalVar.n_features; j++)
                    {
                        batch_input.item_features[i-l, j] = data.item_features[data.test_set[i].item_id].item_feats[j];
                    }
                }
                batch_input.item_seq_features = new DenseTensor<long>(new[] { r-l, GlobalVar.max_seq_len, GlobalVar.n_features });
                // based on batch_input.item_seq
                for (int i=l; i<r; i++)
                {
                    for (int j=0; j<GlobalVar.max_seq_len; j++)
                    {
                        for (int k=0; k<GlobalVar.n_features; k++)
                        {
                            batch_input.item_seq_features[i-l, j, k] = data.item_features[batch_input.item_seq[i-l, j]].item_feats[k];
                        }
                    }
                }
            }
            return batch_input;
        }
    }
}