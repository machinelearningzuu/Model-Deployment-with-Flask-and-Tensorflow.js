async function model(inputs){
    const csvUrl = 'http://127.0.0.1:53669/iris.csv';
    const trainingData = tf.data.csv(csvUrl, {
        columnConfigs: {
            species: {
                isLabel: true
            }
        }
    });

    const numOfFeatures = (await trainingData.columnNames()).length - 1;
    const numOfSamples = 150;
    const convertedData =
          trainingData.map(({xs, ys}) => {
              const labels = [
                    ys.species == "virginica" ? 1 : 0,   
                    ys.species == "versicolor" ? 1 : 0,
                    ys.species == "setosa" ? 1 : 0
              ]
              return{ xs: Object.values(xs), ys: Object.values(labels)};
          }).batch(1);

    const model = tf.sequential();
    model.add(tf.layers.dense({inputShape: [numOfFeatures], activation: "relu", units: 32}))
    model.add(tf.layers.dense({activation: "relu", units: 16}))
    model.add(tf.layers.dense({activation: "softmax", units: 3}));

    model.compile({
        loss: "categoricalCrossentropy", 
        optimizer: tf.train.sgd(0.005),
        metrics: ['accuracy']
                });

    await model.fitDataset(convertedData,
                     {epochs:50,
                      callbacks:{
                          onEpochEnd: async(epoch, logs) =>{
                              console.log("Epoch: " + epoch + " Loss: " + logs.loss + " Accuracy: " + logs.acc);
                          }
                      }});

    const prediction = model.predict(inputs);
    const P = tf.argMax(prediction, axis=1).dataSync();

    const classes = ["Setosa", "Virginica", "Versicolor"];
    alert(classes[p])

}
const input1 = tf.tensor2d([6.4, 3.2, 4.5, 1.5], [1, 4]);
//const input2 = tf.tensor2d([5.8,2.7,5.1,1.9], [1, 4]);
//const input3 = tf.tensor2d([4.4, 2.9, 1.4, 0.2], [1, 4]);
model(input1);