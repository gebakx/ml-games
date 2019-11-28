using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using System;

public class naiveBayes : MonoBehaviour
{
    // data
    string[][] x_train =
    {
        new string[] {"convex", "brown", "narrow", "black"},
        new string[] {"convex", "yellow", "broad", "black"},
        new string[] {"bell", "white", "broad", "brown"},
        new string[] {"convex", "white", "narrow", "brown"},
        new string[] {"convex", "yellow", "broad", "brown"},
        new string[] {"bell", "white", "broad", "brown"},
        new string[] {"convex", "white", "narrow", "pink"}
            };
    string[] y_train = {"poisonous", "edible", "edible", "poisonous", "edible", "edible", "poisonous"};

    string[] test = { "convex", "brown", "narrow", "black" };

    string[] features = { "cap-shape", "cap-color", "gill-size", "gill-color" };

    // model
    int N;
    Dictionary<string, int> Ny;
    Dictionary<string, Dictionary<string, Dictionary<string,int>>> Nxiy;


    void Start()
    {
        learn();
        showModel();
        predict();
    }

    void learn()
    {
        Func<int, List<string>> getColumn = 
            (a) => Enumerable.Range(a, x_train.Length).Select(
                i => x_train[i][a]
                ).ToList();
        Func<List<string>, Dictionary<string, int>> getCount = 
            (l) => l.GroupBy(
                x => x
                ).ToDictionary(
                    g => g.Key, 
                    g => g.Count()
                    );
        Func<string, int, List<string>> filter =
            (cl, at) => Enumerable.Range(0, y_train.Length).Where(
                i => y_train[i] == cl
                ).Select(
                    i => x_train[i][at]
                    ).ToList();
 
        N = x_train.Length;
        Ny = new Dictionary<string, int>();
        Ny = getCount(y_train.ToList());

        // y -> atr -> val -> N
        Nxiy = new Dictionary<string, Dictionary<string, Dictionary<string, int>>>();
        Nxiy = Ny.Keys.ToDictionary(
            cl => cl,
            cl => Enumerable.Range(0, features.Length).ToDictionary(
                i => features[i],
                i => getCount(filter(cl, i))
            )
        );
    }

    void predict()
    {
        Func<int, int, double> laplace =
            (x, y) => (x + 1.0) / (y + N);

        Func<int, int, double> normal =
            (x, y) => (double)x / y;

        Func<int, int, double> actual = laplace;    

        Func<string, double> probability = 
            (cl) => actual(Ny[cl], N) *
            Enumerable.Range(0, features.Length).Select(
                i => Nxiy[cl][features[i]].ContainsKey(test[i]) ? 
                    actual(Nxiy[cl][features[i]][test[i]], Ny[cl]) : 
                    actual(0, Ny[cl])
                ).Aggregate(
                    (x, y) => x * y);

        string pred = Ny.Keys.Select(
            cl => (probability(cl), cl)
            ).Max().Item2;

        Debug.Log("class(" + string.Join(", ", test) + "): " + pred);
    }

    void showModel()
    {
        string s = "Model:\n";
        s += "N: " + N.ToString() + "\n";
        s += "Ny:\n";
        foreach (var pair in Ny)
            s += "  " + pair.Key + ": " + pair.Value.ToString() + "\n";
        s += "Nxiy:\n";
        foreach (var cl in Nxiy)
        {
            s += "  " + cl.Key + "\n";
            foreach (var atr in cl.Value)
            {
                s += "    " + atr.Key + "\n";
                foreach (var val in atr.Value)
                {
                    s += "      " + val.Key + ": " + val.Value.ToString() + "\n";
                }
            }
        }
        Debug.Log(s);
    }
}
