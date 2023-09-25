// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

using System;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Generic;
using System.IO;


namespace inference_csharp
{

    public static class GlobalVar
    {
        public static int n_users = 940;
        public static int n_items = 1017;
        public static int embedding_size = 64;
        public static int max_seq_len = 10;
        public static int batch_size = 512;
        public static bool has_feature = false;
        public static int n_features = 2;
        public static List<string> useful_names = new List<string> { "item_id", "item_seq"};
    }
    class Program
    {
        static void Main(string[] args)
        {
            var history_file = "path/to/user_history.csv";
            var feature_file = "path/to/item_features.csv";
            var test_file = "path/to/test.csv";
            var modelPath = "path/to/model.onnx";
            var output_dir = "path/to/output";
            var task_type = "score"; //score, user embedding, item embedding
            Predict_once(modelPath, history_file, feature_file, test_file, output_dir, task_type);
        }

        static void Predict_once(string modelPath, string history_file, string feature_file, string test_file, string output_dir, string task_type)
        {
            Console.WriteLine("Loading data and model weights...");
            Data data = Utilities.LoadData(history_file, feature_file, test_file);
            // Item_Embedding item_embeddings = Utilities.LoadEmbedding(item_embedding_file);
            var session = new InferenceSession(modelPath); //Load onnx model

            Console.WriteLine("Predicting...");
            int test_length = data.test_set.Count;
            //all model output
            float[,] user_emb_out = new float[test_length, GlobalVar.embedding_size];
            float[,] item_emb_out = new float[test_length, GlobalVar.embedding_size];
            float[] score_out = new float[test_length];
            for (int start = 0; start < test_length; start = start + GlobalVar.batch_size)
            {
                int end = Math.Min(start + GlobalVar.batch_size, test_length);
                Console.WriteLine("Predicting " + start + " to " + end);
                Batch_Input input = Utilities.collate_batch(data, start, end);

                //onnx model inference for one batch, if batch_size=1, then compute only one user embedding
                // based on useful_names, create a list of NamedOnnxValue
                var onnx_input = new List<NamedOnnxValue>();
                foreach (var name in GlobalVar.useful_names)
                {
                    if (name == "user_id")
                    {
                        onnx_input.Add(NamedOnnxValue.CreateFromTensor<long>(name, input.user_id));
                    }
                    else if (name == "item_id")
                    {
                        onnx_input.Add(NamedOnnxValue.CreateFromTensor<long>(name, input.item_id));
                    }
                    else if (name == "item_features")
                    {
                        onnx_input.Add(NamedOnnxValue.CreateFromTensor<long>(name, input.item_features));
                    }
                    else if (name == "item_seq")
                    {
                        onnx_input.Add(NamedOnnxValue.CreateFromTensor<long>(name, input.item_seq));
                    }
                    else if (name == "item_seq_len")
                    {
                        onnx_input.Add(NamedOnnxValue.CreateFromTensor<long>(name, input.item_seq_len));
                    }
                    else if (name == "item_seq_features")
                    {
                        onnx_input.Add(NamedOnnxValue.CreateFromTensor<long>(name, input.item_seq_features));
                    }
                    else if (name == "time_seq")
                    {
                        onnx_input.Add(NamedOnnxValue.CreateFromTensor<long>(name, input.time_seq));
                    }
                }

                // var onnx_input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<long>("item_seq", input.item_seq), NamedOnnxValue.CreateFromTensor<long>("item_seq_len", input.item_seq_len), NamedOnnxValue.CreateFromTensor<long>("time_seq", input.time_seq) };
                var onnx_output = session.Run(onnx_input);

                //print output
                foreach (var item in onnx_output)
                {
                    if (item.Name == "scores")
                    {
                        var tensor = item.AsTensor<float>();
                        for (int i=start; i<end; i++)
                        {
                            score_out[i] = tensor[i-start];
                        }
                    }
                    else if (item.Name == "item_embedding")
                    {
                        var tensor = item.AsTensor<float>();
                        for (int i = start; i < end; i++)
                        {
                            for (int j = 0; j < GlobalVar.embedding_size; j++)
                            {
                                item_emb_out[i, j] = tensor[i - start, j];
                            }
                        }
                    }
                    else if (item.Name == "user_embedding")
                    {
                        var tensor = item.AsTensor<float>();
                        for (int i = start; i < end; i++)
                        {
                            for (int j = 0; j < GlobalVar.embedding_size; j++)
                            {
                                user_emb_out[i, j] = tensor[i - start, j];
                            }
                        }

                    }
                    // Console.WriteLine(item.Name);
                    // Console.WriteLine(item.AsTensor<float>().GetArrayString());
                }
            }
            
            Console.WriteLine("Writing results to file...");

            if (task_type == "score")
            {
                var output_score_file = output_dir + "score.txt";
                FileStream fs = new FileStream(output_score_file, FileMode.Create);
                StreamWriter sw = new StreamWriter(fs);
                foreach (var score in score_out)
                {
                    sw.Write(Convert.ToString(score) + "\n");
                    sw.Flush();
                }
                sw.Close();
                fs.Close();
            }
            else if (task_type == "user embedding")
            {
                var output_user_emb_file = output_dir + "user_embedding.txt";
                FileStream fs = new FileStream(output_user_emb_file, FileMode.Create);
                StreamWriter sw = new StreamWriter(fs);
                for (int i = 0; i < test_length; i++)
                {
                    for (int j = 0; j < GlobalVar.embedding_size; j++)
                    {
                        sw.Write(Convert.ToString(user_emb_out[i, j]) + " ");
                    }
                    sw.Write("\n");
                    sw.Flush();
                }
                sw.Close();
                fs.Close();
            }
            else if (task_type == "item embedding")
            {
                var output_item_emb_file = output_dir + "item_embedding.txt";
                FileStream fs = new FileStream(output_item_emb_file, FileMode.Create);
                StreamWriter sw = new StreamWriter(fs);
                for (int i = 0; i < test_length; i++)
                {
                    for (int j = 0; j < GlobalVar.embedding_size; j++)
                    {
                        sw.Write(Convert.ToString(item_emb_out[i, j]) + " ");
                    }
                    sw.Write("\n");
                    sw.Flush();
                }
                sw.Close();
                fs.Close();
            }
            else
            {
                Console.WriteLine("Wrong task type!");
            }

            Console.WriteLine("Done!");
        }
    }
}
