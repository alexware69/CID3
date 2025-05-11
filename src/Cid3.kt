import org.apache.commons.cli.*
import java.io.*
import java.time.Duration
import java.time.Instant
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.*
import java.util.zip.GZIPInputStream
import java.util.zip.GZIPOutputStream
import javax.sound.sampled.AudioSystem
import kotlin.math.*
import kotlin.system.exitProcess


class Cid3 : Serializable {
    private val version = "1.3.1"
    private var createdWith = ""
    enum class AttributeType {
        Discrete, Continuous, Ignore
    }
    //The number of attributes including the output attribute
    private var numAttributes = 0
    //The names of all attributes.  It is an array of dimension numAttributes.  The last attribute is the output attribute
    private lateinit var attributeNames: ArrayList<String>
    private lateinit var attributeTypes: ArrayList<AttributeType>
    private var classAttribute = 0
    private lateinit var meanValues: DoubleArray
    private lateinit var mostCommonValues: IntArray
    lateinit var fileName: String
    private var seed: Long = 13579
    private var maxThreads = 500
    @Transient
    var globalThreads = ArrayList<Thread?>()
    private val attributeImportance = ArrayList<Triple<Int, Double, Double>>()
    private lateinit var classProbabilities : Probabilities

    //Possible values for each attribute is stored in the domains Arrays.
    private lateinit var domainsIndexToValue: ArrayList<HashMap<Int, String>>
    private lateinit var domainsValueToIndex: ArrayList<SortedMap<String, Int>>

    private lateinit var falsePositivesTrain: IntArray
    private lateinit var falseNegativesTrain: IntArray

    private lateinit var falsePositivesTest: IntArray
    private lateinit var falseNegativesTest: IntArray

    private lateinit var classNoOfCasesTrain: IntArray
    private lateinit var classNoOfCasesTest: IntArray


    enum class Criteria {
        Certainty
    }

    @Transient
    var testData = ArrayList<DataPoint>()

    @Transient
    var trainData = ArrayList<DataPoint>()

    @Transient
    var crossValidationChunks = ArrayList<ArrayList<DataPoint>>()

    @Transient
    var testDataExists = false

    @Transient
    var splitTrainData = false

    @Transient
    var isRandomForest = false

    @Transient
    var isCrossValidation = false

    var numberOfTrees = 1

    @Transient
    var save = false

    @Transient
    var runAnimationReading = true

    @Transient
    var runAnimationReadingTest = true

    @Transient
    var runAnimationCalculating = true

    @Transient
    var runAnimationAnalysis = true


    //The root of the decomposition tree
    var root = TreeNode()
    var rootsRandomForest = ArrayList<TreeNode>()

    @Transient
    var cvRandomForests = ArrayList<ArrayList<TreeNode>>()

    @Transient
    var rootsCrossValidation = ArrayList<TreeNode>()

    var criteria = Criteria.Certainty

    @Transient
    var totalRules = 0

    @Transient
    var totalNodes = 0

    //This function returns an integer corresponding to the symbolic value of the attribute.
    //If the symbol does not exist in the domain, the symbol is added to the domain of the attribute
    private fun getSymbolValue(attribute: Int, symbol: String): Int {
        return if (domainsValueToIndex[attribute][symbol] != null)
            domainsValueToIndex[attribute][symbol]!!
        else {
            val size = domainsIndexToValue[attribute].size
            domainsIndexToValue[attribute][size] = symbol
            domainsValueToIndex[attribute][symbol] = size
            size
        }
    }

    //Returns the most common class for the specified node
    private fun getMostCommonClass(n: TreeNode?): Int {
        val numValuesClass = domainsIndexToValue[classAttribute].size
        var value = n!!.frequencyClasses[0]
        var result = 0
        for (i in 1 until numValuesClass) {
            if (n.frequencyClasses[i] > value) {
                value = n.frequencyClasses[i]
                result = i
            }
        }
        return result
    }

    //Returns a subset of data, in which the value of the specified attribute of all data points is the specified value
    private fun getSubset(data: ArrayList<DataPoint>, attribute: Int, value: Int): DataFrequencies {
        val subset = ArrayList<DataPoint>()
        val frequencies = IntArray(domainsIndexToValue[classAttribute].size)
        for (point in data) {
            if (point.attributes[attribute] == value) {
                subset.add(point)
                frequencies[point.attributes[classAttribute]]++
            }
        }
        return DataFrequencies(subset, frequencies)
    }

    private fun getFrequencies(data: ArrayList<DataPoint>): IntArray {
        val frequencies = IntArray(domainsIndexToValue[classAttribute].size)
        for (point in data) {
            frequencies[point.attributes[classAttribute]]++
        }
        return frequencies
    }

    private fun getSubsetsBelowAndAbove(data: ArrayList<DataPoint>, attribute: Int, value: Double): Pair<DataFrequencies, DataFrequencies> {
        val subsetBelow = ArrayList<DataPoint>()
        val subsetAbove = ArrayList<DataPoint>()
        val frequenciesBelow = IntArray(domainsIndexToValue[classAttribute].size)
        val frequenciesAbove = IntArray(domainsIndexToValue[classAttribute].size)
        for (point in data) {
            if ((domainsIndexToValue[attribute][point.attributes[attribute]])?.toDouble()!! <= value) {
                subsetBelow.add(point)
                frequenciesBelow[point.attributes[classAttribute]]++
            } else {
                subsetAbove.add(point)
                frequenciesAbove[point.attributes[classAttribute]]++
            }
        }
        return Pair(DataFrequencies(subsetBelow, frequenciesBelow), DataFrequencies(subsetAbove, frequenciesAbove))
    }

    //This is the final form of the certainty function.
    private fun calculateCertainty(data: ArrayList<DataPoint>, givenThatAttribute: Int): Certainty {
        val numData = data.size
        if (numData == 0) return Certainty(0.0, 0.0, null, 0.0)
        val numValuesClass = domainsIndexToValue[classAttribute].size
        val numValuesGivenAtt = domainsIndexToValue[givenThatAttribute].size

        //If attribute is discrete
        return if (attributeTypes[givenThatAttribute] == AttributeType.Discrete) {
            val probabilities = calculateAllProbabilities(data)
            var sum: Double
            var sum2 = 0.0
            var sumClass = 0.0
            var probability: Double
            var probabilityCAndA: Double
            for (j in 0 until numValuesGivenAtt) {
                if (domainsIndexToValue[givenThatAttribute][j] == "?") continue
                probability = probabilities[givenThatAttribute].prob[j]
                sum = 0.0
                for (i in 0 until numValuesClass) {
                    probabilityCAndA = probabilities[givenThatAttribute].probCAndA[j][i]
                    sum += abs(probabilityCAndA - 1.0 * probability / numValuesClass)
                }
                sum2 += sum
            }
            //Calculate class certainty
            for (i in 0 until numValuesClass) {
                probability = probabilities[classAttribute].prob[i]
                sumClass += abs(probability - 1.0 * 1 / numValuesClass)
            }
            Certainty(sum2, 0.0, null, sumClass)
        } else {
            var finalThreshold = 0.0
            var finalThresholdObject: Threshold? = null
            var totalCertainty: Double
            var finalTotalCertainty = 0.0
            var sumClass = 0.0
            var probability: Double
            //---------------------------------------------------------------------------------------------------------
            //Implementation of thresholds using a sorted set
            val attributeValuesSet: SortedSet<Double> = TreeSet()
            val attributeToClass = HashMap<Double?, Pair<Int, Boolean>>()
            for (point in data) {
                val attribute = (domainsIndexToValue[givenThatAttribute][point.attributes[givenThatAttribute]])?.toDouble()
                val theClass = point.attributes[classAttribute]
                attributeValuesSet.add(attribute)
                val pair = attributeToClass[attribute]
                if (pair != null) {
                    if (pair.first != theClass && pair.second) attributeToClass[attribute] = Pair(theClass, false)
                } else attributeToClass[attribute] = Pair(theClass, true)
            }
            val it: Iterator<Double> = attributeValuesSet.iterator()
            var attributeValue = it.next()
            var attributeClass1 = attributeToClass[attributeValue]!!
            var theClass = attributeClass1.first
            val thresholds = ArrayList<Threshold>()
            while (it.hasNext()) {
                val attributeValue2 = it.next()
                val attributeClass2 = attributeToClass[attributeValue2]!!
                val theClass2 = attributeClass2.first
                if (theClass2 != theClass || !attributeClass2.second || !attributeClass1.second) {
                    //Add threshold
                    val median = (attributeValue + attributeValue2) / 2
                    thresholds.add(Threshold(median, arrayOfNulls(numValuesClass)))
                }
                //Set new point
                attributeValue = attributeValue2
                theClass = theClass2
                attributeClass1 = attributeClass2
            }

            //---------------------------------------------------------------------------------------------------------
            //If there are no thresholds return zero.
            if (thresholds.isEmpty()) return Certainty(0.0, 0.0, null, 0.0)

            //This trick reduces the possible thresholds to just ONE or TWO, dramatically improving running times!
            //=========================================================
            val centerThresholdIndex = thresholds.size / 2
            val centerThreshold: Threshold
            val centerThreshold1: Threshold
            when {
                thresholds.size == 1 -> {
                    centerThreshold = thresholds[0]
                    thresholds.clear()
                    thresholds.add(centerThreshold)
                }
                thresholds.size % 2 != 0 -> {
                    centerThreshold = thresholds[centerThresholdIndex]
                    thresholds.clear()
                    thresholds.add(centerThreshold)
                }
                else -> {
                    centerThreshold = thresholds[centerThresholdIndex]
                    centerThreshold1 = thresholds[centerThresholdIndex - 1]
                    thresholds.clear()
                    thresholds.add(centerThreshold)
                    thresholds.add(centerThreshold1)
                }
            }
            //=========================================================
            var probABelow: Double
            var probAAbove: Double
            var probCAndABelow: Double
            var probCAndAAbove: Double
            var certaintyBelow: Double
            var certaintyAbove: Double
            //Loop through the data just one time
            for (point in data) {
                //For each threshold count data to get prob and probC_And_A
                val theClass1 = point.attributes[classAttribute]
                for (iThreshold in thresholds) {
                    if (iThreshold.sumsClassesAndAttribute[theClass1] == null) iThreshold.sumsClassesAndAttribute[theClass1] = SumBelowAndAbove(0, 0)
                    if ((domainsIndexToValue[givenThatAttribute][point.attributes[givenThatAttribute]])?.toDouble()!! <= iThreshold.value) {
                        iThreshold.sumABelow++
                        //Next calculate probability of c and a
                        iThreshold.sumsClassesAndAttribute[theClass1]!!.below++
                    } else {
                        iThreshold.sumAAbove++
                        //Next calculate probability of c and a
                        iThreshold.sumsClassesAndAttribute[theClass1]!!.above++
                    }
                }
            }

            //Now calculate probabilities
            for (threshold in thresholds) {
                //Calculate prob
                probABelow = 1.0 * threshold.sumABelow / numData
                probAAbove = 1.0 * threshold.sumAAbove / numData

                //Reset the certainty
                certaintyBelow = 0.0
                certaintyAbove = 0.0
                for (c in 0 until numValuesClass) {
                    if (threshold.sumsClassesAndAttribute[c] != null) {
                        probCAndABelow = 1.0 * threshold.sumsClassesAndAttribute[c]!!.below / numData
                        probCAndAAbove = 1.0 * threshold.sumsClassesAndAttribute[c]!!.above / numData
                    } else {
                        probCAndABelow = 0.0
                        probCAndAAbove = 0.0
                    }
                    certaintyBelow += abs(probCAndABelow - probABelow / numValuesClass)
                    certaintyAbove += abs(probCAndAAbove - probAAbove / numValuesClass)
                }
                //Calculate totals
                totalCertainty = certaintyBelow + certaintyAbove
                if (finalTotalCertainty < totalCertainty) {
                    finalTotalCertainty = totalCertainty
                    finalThreshold = threshold.value
                    finalThresholdObject = threshold
                }
            }
            //Calculate class certainty
            //Only do for root node
            if (data.size == root.data.size) {
                for (i in 0 until numValuesClass) {
                    probability = classProbabilities.prob[i]
                    sumClass += abs(probability - 1.0 * 1 / numValuesClass)
                }
            }
            Certainty(finalTotalCertainty, finalThreshold, finalThresholdObject, sumClass)
        }
    }

    //This method calculates all probabilities in one run
    private fun calculateAllProbabilities(data: ArrayList<DataPoint>): ArrayList<Probabilities> {
        val numData = data.size
        val probabilities = ArrayList<Probabilities>(numAttributes)

        //Initialize the array
        for (j in 0 until numAttributes) {
            //if (attributeTypes[j] == AttributeType.Ignore) continue
            val p = Probabilities(j,domainsIndexToValue,classAttribute)
            probabilities.add(p)
        }
        //Count occurrences
        for (point in data) {
            for (j in point.attributes.indices) {
                if (attributeTypes[j] == AttributeType.Ignore) continue
                probabilities[j].prob[point.attributes[j]]++
                probabilities[j].probCAndA[point.attributes[j]][point.attributes[classAttribute]]++
            }
        }
        //Divide all values by total data size to get probabilities.
        var current: Probabilities
        for (i in probabilities.indices) {
            if (attributeTypes[i] == AttributeType.Ignore) continue
            current = probabilities[i]
            for (j in current.prob.indices) {
                current.prob[j] = 1.0 * current.prob[j] / numData
            }
            for (j in current.probCAndA.indices) {
                for (k in current.probCAndA[j].indices) {
                    current.probCAndA[j][k] = 1.0 * current.probCAndA[j][k] / numData
                }
            }
            //Calculate ProbC_Given_A
            for (j in current.probCGivenA.indices) {
                for (k in current.probCGivenA[j].indices) {
                    current.probCGivenA[j][k] = 1.0 * current.probCAndA[j][k] / current.prob[j]
                }
            }
        }
        return probabilities
    }

    //This method calculates class probabilities
    private fun calculateClassProbabilities(): Probabilities {
        val numData = root.data.size
        val p = Probabilities(numAttributes - 1, domainsIndexToValue, classAttribute)
        //Count occurrences
        for (point in root.data) {
            p.prob[point.attributes[numAttributes - 1]]++
        }
        //Divide all values by total data size to get probabilities.
        for (j in p.prob.indices) {
            p.prob[j] = p.prob[j] / numData
        }
        return p
    }

    private fun calculateImportanceCertainties(){
        var certainty: Certainty
        for (i in 0 until numAttributes - 1) {
            if (attributeTypes[i] == AttributeType.Ignore) continue
            certainty = calculateCertainty(root.data, i)
            //Insert into attributeImportance
            attributeImportance.add(Triple(i, certainty.certainty, certainty.certaintyClass))
        }
    }

    //This function checks if the specified attribute is used to decompose the data set
    //in any of the parents of the specified node in the decomposition tree.
    //Recursively checks the specified node as well as all parents

    private fun alreadyUsedToDecompose(node: TreeNode?, attribute: Int): Boolean {
        if (node!!.children.isNotEmpty()) {
            if (node.decompositionAttribute == attribute) return true
        }
        return if (node.parent == null) false else alreadyUsedToDecompose(node.parent, attribute)
    }

    private fun stopConditionAllClassesEqual(frequencyClasses: IntArray): Boolean {
        val numValuesClass = domainsIndexToValue[classAttribute].size
        var oneClassIsPresent = false
        for (i in 0 until numValuesClass) {
            if (frequencyClasses[i] != 0) {
                oneClassIsPresent = if (!oneClassIsPresent) true else return false
            }
        }
        return true
    }

    //This function decomposes the specified node according to the id3 algorithm.
    //Recursively divides all children nodes until it is not possible to divide any further
    private fun decomposeNode(node: TreeNode, selectedAttributes: ArrayList<Int>, mySeed: Long) {
        var selectedAttributesLocal = selectedAttributes
        var bestCertainty = Certainty(0.0, 0.0, null, 0.0)
        var selected = false
        var selectedAttribute = 0
        if (criteria == Criteria.Certainty) {
            if (node.data.size <= 1) return
            if (stopConditionAllClassesEqual(node.frequencyClasses)) return
            //In the following loop, the best attribute is located
            var certainty: Certainty
            for (selectedAtt in selectedAttributesLocal) {
                if (classAttribute == selectedAtt) continue
                if (attributeTypes[selectedAtt] == AttributeType.Discrete && alreadyUsedToDecompose(node, selectedAtt)) continue
                certainty = calculateCertainty(node.data, selectedAtt)
                if (certainty.certainty == 0.0) continue
                //Insert into attributeImportance
                if(node.parent == null && !isRandomForest){
                    attributeImportance.add(Triple(selectedAtt, certainty.certainty, certainty.certaintyClass))
                }
                //Select best attribute
                if (certainty.certainty > bestCertainty.certainty) {
                    selected = true
                    bestCertainty = certainty
                    selectedAttribute = selectedAtt
                }
            }
            if (!selected || bestCertainty.certainty == 0.0) return
        }
        node.certaintyUsedToDecompose = bestCertainty.certainty

        //if attribute is discrete
        if (attributeTypes[selectedAttribute] == AttributeType.Discrete) {
            //Now divide the dataset using the selected attribute
            val numValues = domainsIndexToValue[selectedAttribute].size
            node.decompositionAttribute = selectedAttribute
            var df: DataFrequencies
            for (j in 0 until numValues) {
                if (domainsIndexToValue[selectedAttribute][j] == null || domainsIndexToValue[selectedAttribute][j] == "?") continue
                val newNode = TreeNode()
                newNode.parent = node
                df = getSubset(node.data, selectedAttribute, j)
                newNode.data = df.data
                newNode.frequencyClasses = df.frequencyClasses
                newNode.decompositionValue = j
                node.children.add(newNode)
            }
            //Recursively divides children nodes
            when {
                isCrossValidation -> {
                    for (j in node.children.indices) {
                        decomposeNode(node.children[j], selectedAttributesLocal, 0)
                    }
                }
                isRandomForest -> {
                    val rand = Random(mySeed)
                    var randomAttribute: Int
                    val numAtt = selectedAttributesLocal.size
                    for (j in node.children.indices) {
                        selectedAttributesLocal = ArrayList()
                        while (selectedAttributesLocal.size < numAtt) {
                            randomAttribute = rand.nextInt(numAttributes - 1)
                            if (!selectedAttributesLocal.contains(randomAttribute) && attributeTypes[randomAttribute] != AttributeType.Ignore)
                                selectedAttributesLocal.add(randomAttribute)
                        }
                        decomposeNode(node.children[j], selectedAttributesLocal, mySeed + 1 + j)
                    }
                }
                else -> {
                    for (j in 0 until node.children.size) {
                        val selectedAttributesLocal2: ArrayList<Int> = selectedAttributesLocal
                        if (globalThreads.size < maxThreads) {
                            val thread = Thread {decomposeNode(node.children[j], selectedAttributesLocal2, mySeed + 1 + j) }
                            thread.start()
                            globalThreads.add(thread)
                        } else decomposeNode(node.children[j], selectedAttributesLocal2, mySeed + 1 + j)
                    }
                }
            }
        }
        //If attribute is continuous
        else {
            node.decompositionAttribute = selectedAttribute
            var df: DataFrequencies
            //First less than threshold
            var newNode = TreeNode()
            newNode.parent = node
            val subsets = getSubsetsBelowAndAbove(node.data, selectedAttribute, bestCertainty.threshold)
            df = subsets.first
            newNode.data = df.data
            newNode.frequencyClasses = df.frequencyClasses
            newNode.decompositionValueContinuous = " <= " + bestCertainty.threshold
            newNode.decompositionValueContinuousSymbol = "<="
            newNode.thresholdContinuous = bestCertainty.threshold
            node.children.add(newNode)

            //Then over the threshold.
            newNode = TreeNode()
            newNode.parent = node
            df = subsets.second
            newNode.data = df.data
            newNode.frequencyClasses = df.frequencyClasses
            newNode.decompositionValueContinuous = " > " + bestCertainty.threshold
            newNode.decompositionValueContinuousSymbol = ">"
            newNode.thresholdContinuous = bestCertainty.threshold
            node.children.add(newNode)

            //Decompose children
            if (node.children[0].data.isEmpty() || node.children[1].data.isEmpty()) return
            when {
                isCrossValidation -> {
                    decomposeNode(node.children[0], selectedAttributesLocal, 0)
                    decomposeNode(node.children[1], selectedAttributesLocal, 0)
                }
                isRandomForest -> {
                    val rand = Random(mySeed)
                    var randomAttribute: Int
                    val numAtt = selectedAttributesLocal.size
                    selectedAttributesLocal = ArrayList()
                    while (selectedAttributesLocal.size < numAtt) {
                        randomAttribute = rand.nextInt(numAttributes - 1)
                        if (!selectedAttributesLocal.contains(randomAttribute) && attributeTypes[randomAttribute] != AttributeType.Ignore)
                            selectedAttributesLocal.add(randomAttribute)
                    }
                    decomposeNode(node.children[0], selectedAttributesLocal, mySeed + 1)
                    selectedAttributesLocal = ArrayList()
                    while (selectedAttributesLocal.size < numAtt) {
                        randomAttribute = rand.nextInt(numAttributes - 1)
                        if (!selectedAttributesLocal.contains(randomAttribute) && attributeTypes[randomAttribute] != AttributeType.Ignore)
                            selectedAttributesLocal.add(randomAttribute)
                    }
                    decomposeNode(node.children[1], selectedAttributesLocal, mySeed + 2)
                }
                else -> {
                    val selectedAttributesLocal2: ArrayList<Int> = selectedAttributesLocal
                    if (globalThreads.size < maxThreads) {
                        val thread = Thread {decomposeNode(node.children[0], selectedAttributesLocal2, mySeed + 1)}
                        thread.start()
                        globalThreads.add(thread)
                    } else decomposeNode(node.children[0], selectedAttributesLocal2, mySeed + 1)

                    if (globalThreads.size < maxThreads) {
                        val thread = Thread {decomposeNode(node.children[1], selectedAttributesLocal2, mySeed + 2)}
                        thread.start()
                        globalThreads.add(thread)
                    } else decomposeNode(node.children[1], selectedAttributesLocal2, mySeed + 2)

                    //decomposeNode(node.children!![0], selectedAttributesLocal, mySeed + 1)
                    //decomposeNode(node.children!![1], selectedAttributesLocal, mySeed + 2)
                }
            }
        }
    }

    fun imputeMissing() {
        for (attribute in 0 until numAttributes - 1) {
            if (attributeTypes[attribute] == AttributeType.Continuous) {
                if (domainsIndexToValue[attribute].containsValue("?")) {
                    //Find mean value
                    val mean = meanValues[attribute]
                    //Get index
                    val index = domainsValueToIndex[attribute]["?"]!!
                    //Replace missing with mean
                    domainsIndexToValue[attribute].replace(index, "?", mean.toString())
                    domainsValueToIndex[attribute].remove("?")
                    domainsValueToIndex[attribute][mean.toString()] = index
                }
            }
        }
        for (point in trainData){
            for (i in 0 until point.attributes.size - 1){
                if (point.attributes[i] == domainsValueToIndex[i]["?"]){
                    if (attributeTypes[i] == AttributeType.Discrete)
                        point.attributes[i] = mostCommonValues[i]
                }
            }
        }
        if (testDataExists) {
            for (point in testData) {
                for (i in 0 until point.attributes.size - 1) {
                    if (point.attributes[i] == domainsValueToIndex[i]["?"]) {
                        if (attributeTypes[i] == AttributeType.Discrete)
                            point.attributes[i] = mostCommonValues[i]
                    }
                }
            }
        }
    }

    //Find the mean value of a continuous attribute
    private fun getMeanValue(attribute: Int): Double {
        var sum = 0.0
        var counter = 0
        for (point in trainData) {
            try {
                val attValue = (domainsIndexToValue[attribute][point.attributes[attribute]])?.toDouble()
                if (attValue != null) {
                    sum += attValue
                }
                counter++
            } catch (_: Exception) {
                //continue;
            }
        }
        return sum / counter
    }

    fun setMeanValues() {
        meanValues = DoubleArray(numAttributes - 1)
        for (i in 0 until numAttributes - 1) {
            if (attributeTypes[i] == AttributeType.Ignore) continue
            if (attributeTypes[i] == AttributeType.Continuous) {
                meanValues[i] = getMeanValue(i)
            } else meanValues[i] = 0.0
        }
    }

    //Find the most common values of a discrete attribute. This is needed for imputation.
    private fun getMostCommonValue(attribute: Int): Int {
        val frequencies = IntArray(domainsIndexToValue[attribute].size)
        for (point in trainData) {
            frequencies[point.attributes[attribute]]++
        }
        var mostFrequent = 0
        var index = 0
        for (i in frequencies.indices) {
            if (domainsIndexToValue[attribute][i] != "?") if (frequencies[i] > mostFrequent) {
                mostFrequent = frequencies[i]
                index = i
            }
        }
        return index
    }

    fun setMostCommonValues() {
        mostCommonValues = IntArray(numAttributes - 1)
        for (i in 0 until numAttributes - 1) {
            if (attributeTypes[i] == AttributeType.Ignore) continue
            if (attributeTypes[i] == AttributeType.Discrete) {
                mostCommonValues[i] = getMostCommonValue(i)
            } else mostCommonValues[i] = 0
        }
    }

    fun readTestData(filename: String) {
        //Read the test file
        val `in`: FileInputStream?
        val data = ArrayList<DataPoint>()
        try {
            val inputFile = File(filename)
            `in` = FileInputStream(inputFile)
        } catch (e: Exception) {
            System.err.println("Unable to open data file: $filename\n$e")
            exitProcess(1)
        }
        val bin = BufferedReader(InputStreamReader(`in`))
        var input: String?
        var currentLine = 0

        try {
            input = bin.readLine()
            currentLine++
        } catch (_: Exception) {
            System.err.println("Unable to read first line from test file.")
            exitProcess(1)
        }
        if (input == null) {
            System.err.println("No data found in test file: $filename\n")
            exitProcess(1)
        }

        var tokenizer: StringTokenizer
        while (input != null) {
            if (!input.startsWith("|") && !input.startsWith("//") && input.trim() != "") {
                if (input.endsWith('.')) input = input.dropLast(1)
                tokenizer = StringTokenizer(input, ",")
                val numTokens = tokenizer.countTokens()
                if (numTokens != numAttributes) {
                    System.err.println("Read " + data.size + " data")
                    System.err.println("Last line read, #$currentLine: $input")
                    System.err.println("Expecting $numAttributes attributes")
                    exitProcess(1)
                }
                val point = DataPoint(numAttributes)
                var next: String
                var currentColumn: Int
                for (i in 0 until numAttributes) {
                    currentColumn = i + 1
                    next = tokenizer.nextToken().trim { it <= ' ' }
                    if (attributeTypes[i] == AttributeType.Continuous) {
                        if (next == "?" || next == "NaN") point.attributes[i] = getSymbolValue(i, "?") else {
                            try {
                                next.toDouble()
                                point.attributes[i] = getSymbolValue(i, next)
                            } catch (_: Exception) {
                                System.err.println("Error reading continuous value in test data at line #$currentLine, column #$currentColumn.")
                                exitProcess(1)
                            }
                        }
                    } else if (attributeTypes[i] == AttributeType.Discrete) {
                        point.attributes[i] = getSymbolValue(i, next)
                    } else if (attributeTypes[i] == AttributeType.Ignore) {
                        point.attributes[i] = getSymbolValue(i, next)
                    }
                }
                data.add(point)

                try {
                    input = bin.readLine()
                    currentLine++
                } catch (_: Exception) {
                    System.err.println("Unable to read line #$currentLine from test file.")
                    exitProcess(1)
                }
            }
        }
        try {
            bin.close()
        } catch (_: Exception) {
            System.err.println("Unable to close test file.")
            exitProcess(1)
        }
        testData = data

        //Resize root.frequencyClasses in case new class values were found in test dataset
        if (root.frequencyClasses.size < domainsIndexToValue[classAttribute].size) {
            val newArray = IntArray(domainsIndexToValue[classAttribute].size)
            System.arraycopy(root.frequencyClasses, 0, newArray, 0, root.frequencyClasses.size)
            root.frequencyClasses = newArray
        }

        //Stop the animation
        runAnimationReadingTest = false

        print("\r" + "[ ✓ ] " + "Read data: " + testData.size + " cases for testing. ")
        print("\n")
    }

    //Function to read the data file.
    //If any line starts with // it is taken as a comment and ignored.
    //Blank lines are also ignored.
    fun readData(filename: String) {
        val `in`: FileInputStream
        val data = ArrayList<DataPoint>()
        val numTraining: Int
        try {
            val inputFile = File(filename)
            `in` = FileInputStream(inputFile)
        } catch (_: Exception) {
            System.err.println("Unable to open data file: $filename\n")
            exitProcess(1)
        }
        val bin = BufferedReader(InputStreamReader(`in`))
        var input: String?
        var tokenizer: StringTokenizer
        var currentLine = 0

        try {
            input = bin.readLine()
            currentLine++
        } catch (_: Exception) {
            System.err.println("Unable to read first line from data file.")
            exitProcess(1)
        }
        if (input == null) {
            System.err.println("No data found in data file: $filename\n")
            exitProcess(1)
        }
        while (input != null) {
            if (!input.startsWith("|") && !input.startsWith("//") && input.trim() != "") {
                if (input.endsWith('.')) input = input.dropLast(1)
                tokenizer = StringTokenizer(input, ",")
                val numTokens = tokenizer.countTokens()
                if (numTokens != numAttributes) {
                    System.err.println("Read " + data.size + " data")
                    System.err.println("Last line read, #$currentLine: $input")
                    System.err.println("Expecting $numAttributes attributes")
                    exitProcess(1)
                }

                val point = DataPoint(numAttributes)
                var next: String
                var currentColumn: Int
                for (i in 0 until numAttributes) {
                    currentColumn = i + 1
                    next = tokenizer.nextToken().trim { it <= ' ' }
                    if (attributeTypes[i] == AttributeType.Continuous) {
                        if (next == "?" || next == "NaN") point.attributes[i] = getSymbolValue(i, "?")
                        else {
                            try {
                                next.toDouble()
                                point.attributes[i] = getSymbolValue(i, next)
                            } catch (_: Exception) {
                                System.err.println("Error reading continuous value in train data at line #$currentLine, column #$currentColumn.")
                                exitProcess(1)
                            }
                        }
                    } else if (attributeTypes[i] == AttributeType.Discrete) {
                        point.attributes[i] = getSymbolValue(i, next)
                    } else if (attributeTypes[i] == AttributeType.Ignore) {
                        point.attributes[i] = getSymbolValue(i, next)
                    }
                }
                data.add(point)
            }
            try {
                input = bin.readLine()
                currentLine++
            } catch (_: Exception) {
                System.err.println("Unable to read line #$currentLine from data file.")
                exitProcess(1)
            }
        }
        try {
            bin.close()
        } catch (_: Exception) {
            System.err.println("Unable to close data file.")
            exitProcess(1)
        }
        val size = data.size
        root.frequencyClasses = IntArray(domainsIndexToValue[classAttribute].size)
        if (splitTrainData && !testDataExists && !isCrossValidation) {
            //Randomize the data
            data.shuffle()
            numTraining = size * 80 / 100
            for (i in 0 until size) {
                if (i < numTraining) {
                    val point = data[i]
                    root.data.add(point)
                    root.frequencyClasses[point.attributes[classAttribute]]++
                } else testData.add(data[i])
            }
        } else {
            for (point in data) {
                root.data.add(point)
                root.frequencyClasses[point.attributes[classAttribute]]++
            }
        }
        trainData = root.data
        val na = numAttributes - 1

        //Stop the animation
        runAnimationReading = false

        if (splitTrainData && !testDataExists) {
            print("\r" + "[ ✓ ] " + "Read data: " + root.data.size + " cases for training. ($na attributes)")
            print("\n")
            print("\r" + "[ ✓ ] " + "Read data: " + testData.size + " cases for testing.")
        } else {
            print("\r" + "[ ✓ ] " + "Read data: " + root.data.size + " cases for training. ($na attributes)")
        }
        print("\n")
    } //End of function readData

    private fun readNames(filename: String) {
        val `in`: FileInputStream?
        var input: String?
        val attributes = ArrayList<Pair<String, String>>()
        //Read the names file
        try {
            `in` = if (filename.endsWith(".data")) {
                val split = filename.split("\\.data".toRegex()).toTypedArray()
                val inputFile = File(split[0] + ".names")
                FileInputStream(inputFile)
            } else {
                val inputFile = File("$fileName.names")
                FileInputStream(inputFile)
            }
        } catch (_: Exception) {
            System.err.println("Unable to open names file.")
            exitProcess(1)
        }
        val bin = BufferedReader(InputStreamReader(`in`))

        //Read first line containing class values.
        var lineNumber = 0
        try {
            bin.readLine()
            lineNumber++
        } catch (_: Exception) {
            System.err.println("Unable to read first line in names file.")
            exitProcess(1)
        }

        //Save attribute names and types to a pair array.
        try {
            input = bin.readLine()
            lineNumber++
        } catch (_: Exception) {
            val ln = lineNumber + 1
            System.err.println("Unable to read line #$ln in names file.")
            exitProcess(1)
        }
        while (input != null) {
            if (!input.startsWith("|") && !input.startsWith("//") && input.trim() != "") {
                val split = input.split(":".toRegex()).toTypedArray()
                if (split.size == 2) {
                    val t = Pair(split[0].trim { it <= ' ' }, split[1].trim { it <= ' ' })
                    attributes.add(t)
                }
                else {
                    System.err.println("Unable to read attribute type at line #$lineNumber in names file.")
                    exitProcess(1)
                }
            }
            try {
                input = bin.readLine()
                lineNumber++
            } catch (_: Exception) {
                val ln = lineNumber + 1
                System.err.println("Unable to read line #$ln in names file.")
                exitProcess(1)
            }
        }

        //Set numAttributes. +1 for the class attribute
        numAttributes = attributes.size + 1

        //Set class attribute
        classAttribute = numAttributes - 1

        //Check for errors.
        if (numAttributes <= 1) {
            System.err.println("Expecting at least one input attribute and one output attribute")
            exitProcess(1)
        }

        //Initialize domains
        domainsIndexToValue = ArrayList()
        domainsValueToIndex = ArrayList()
        //domains = new ArrayList[numAttributes];
        repeat(numAttributes) {
            domainsIndexToValue.add(HashMap())
        }
        val comparator = NaturalOrderComparator()
        for (i in 0 until numAttributes) {
            if (i < numAttributes - 1)
                domainsValueToIndex.add(TreeMap())
            else domainsValueToIndex.add(TreeMap(comparator))
        }

        //Set attributeNames. They should be in the same order as they appear in the data. +1 for the class
        attributeNames = ArrayList(numAttributes)
        for (i in 0 until numAttributes - 1) {
            val t = attributes[i]
            attributeNames.add(t.first)
        }

        //Set the class. For now all class attribute names are the same: Class.
        attributeNames.add(numAttributes - 1, "Class")

        //Initialize attributeTypes.
        attributeTypes = ArrayList(numAttributes)

        //Set the attribute types.
        for (i in 0 until numAttributes - 1) {
            val attribute = attributes[i]
            if (attribute.second.endsWith(".")) {
                when (attribute.second.trim { it <= ' ' }) {
                    "continuous." -> attributeTypes.add(AttributeType.Continuous)
                    "ignore." -> attributeTypes.add(AttributeType.Ignore)
                    else -> attributeTypes.add(AttributeType.Discrete)
                }
            }
            else{
                System.err.println("All lines in the names file must end with a dot.")
                exitProcess(1)
            }
        }

        //Set attribute type for the class.
        attributeTypes.add(numAttributes - 1, AttributeType.Discrete)
    }

    //  This function counts the total nodes and the leaf nodes
    private fun countNodes(node: TreeNode) {
        if (node.data.isNotEmpty()) totalNodes++
        if (node.children.isEmpty()) {
            if (node.data.isNotEmpty()) totalRules++
            return
        }
        val numValues = node.children.size
        for (i in 0 until numValues) {
            countNodes(node.children[i])
        }
    }

    private fun countClasses() {
        //First cleanup
        for (i in classNoOfCasesTrain.indices){
            classNoOfCasesTrain[i] = 0
        }
        for (i in classNoOfCasesTest.indices){
            classNoOfCasesTest[i] = 0
        }
        var currentClass : Int
        for (point in trainData) {
            currentClass = point.attributes[classAttribute]
            classNoOfCasesTrain[currentClass]++
        }
        var currentClass2 : Int
        for (point in testData) {
            currentClass2 = point.attributes[classAttribute]
            classNoOfCasesTest[currentClass2]++
        }
    }

    private fun createFile() {
        val oos: ObjectOutputStream
        val fOut: FileOutputStream
        var fName = fileName
        fName = fName.substring(0, fName.length - 4)
        fName += "tree"
        try {
            //Check if file exists...delete it
            val inputFile = File(fName)
            if (inputFile.exists()) {
                val res = inputFile.delete()
                if (!res) {
                    print("Error deleting previous tree file.")
                    print("\n")
                    exitProcess(1)
                }
            }

            //Set version, date and time
            val dayInMonth = LocalDateTime.now().dayOfMonth
            var day = LocalDateTime.now().dayOfWeek.name.lowercase(Locale.getDefault())
            day = day.substring(0, 1).uppercase(Locale.getDefault()) + day.substring(1)
            var month = LocalDateTime.now().month.name.lowercase(Locale.getDefault())
            month = month.substring(0, 1).uppercase(Locale.getDefault()) + month.substring(1)
            val time = DateTimeFormatter.ofPattern("hh:mm:ss a")
            val timeString = LocalDateTime.now().format(time)
            val year = LocalDateTime.now().year.toString()
            createdWith = "Created with: CID3 [Version ${version}] on $day $month $dayInMonth, $year $timeString"

            //Serialize and save to disk
            fOut = FileOutputStream(fName, false)
            val gz = GZIPOutputStream(fOut)
            oos = ObjectOutputStream(gz)
            oos.writeObject(this)
            oos.close()
        } catch (ex: Exception) {
            ex.printStackTrace()
        }
    }

    private fun createFileRF() {
        val oos: ObjectOutputStream
        val fOut: FileOutputStream
        var fName = fileName
        fName = fName.substring(0, fName.length - 4)
        fName += "forest"
        try {
            //Check if file exists...delete it
            val inputFile = File(fName)
            if (inputFile.exists()) {
                val res = inputFile.delete()
                if (!res) {
                    print("Error deleting previous tree file.")
                    print("\n")
                    exitProcess(1)
                }
            }

            //Set version, date and time
            val dayInMonth = LocalDateTime.now().dayOfMonth
            var day = LocalDateTime.now().dayOfWeek.name.lowercase(Locale.getDefault())
            day = day.substring(0, 1).uppercase(Locale.getDefault()) + day.substring(1)
            var month = LocalDateTime.now().month.name.lowercase(Locale.getDefault())
            month = month.substring(0, 1).uppercase(Locale.getDefault()) + month.substring(1)
            val time = DateTimeFormatter.ofPattern("hh:mm:ss a")
            val timeString = LocalDateTime.now().format(time)
            val year = LocalDateTime.now().year.toString()
            this.createdWith = "Created with: CID3 [Version ${this.version}] on $day $month $dayInMonth, $year $timeString"

            //Serialize and save to disk
            fOut = FileOutputStream(fName, false)
            val gz = GZIPOutputStream(fOut)
            oos = ObjectOutputStream(gz)
            oos.writeObject(this)
            oos.close()
        } catch (ex: Exception) {
            ex.printStackTrace()
        }
    }

    private fun deserializeFile(file: String): Cid3 {
        val ret: Cid3
        val objectInputStream: ObjectInputStream
        try {
            val `is` = GZIPInputStream(FileInputStream(file))
            objectInputStream = ObjectInputStream(`is`)
            ret = objectInputStream.readObject() as Cid3
            objectInputStream.close()
        } catch (_: Exception) {
            print("Error deserializing file.")
            print("\n")
            exitProcess(1)
        }
        return ret
    }
    private fun calculateCertCGivenAContinuous(data: ArrayList<DataPoint>,threshold: Threshold) : Pair<Double, Double>{
        val numData = data.size
        val totalCertainty: Double
        var sumClass = 0.0
        var probability: Double
        val numValuesClass = domainsIndexToValue[classAttribute].size
        var probCAndABelow: Double
        var probCAndAAbove: Double

        //Now calculate probabilities
        val probABelow: Double = 1.0 * threshold.sumABelow / numData
        val probAAbove: Double = 1.0 * threshold.sumAAbove / numData

        //Reset the certainty
        var certaintyBelow = 0.0
        var certaintyAbove = 0.0
        for (c in 0 until numValuesClass) {
            if (threshold.sumsClassesAndAttribute[c] != null) {
                probCAndABelow = 1.0 * threshold.sumsClassesAndAttribute[c]!!.below / numData
                probCAndAAbove = 1.0 * threshold.sumsClassesAndAttribute[c]!!.above / numData
            } else {
                probCAndABelow = 0.0
                probCAndAAbove = 0.0
            }
            certaintyBelow += abs(probCAndABelow - probABelow / numValuesClass)
            certaintyAbove += abs(probCAndAAbove - probAAbove / numValuesClass)
        }
        //Calculate totals
        totalCertainty = certaintyBelow + certaintyAbove

        //Calculate class certainty
        //Only do for root node
        if (data.size == root.data.size) {
            for (i in 0 until numValuesClass) {
                probability = classProbabilities.prob[i]
                sumClass += abs(probability - 1.0 * 1 / numValuesClass)
            }
        }
        return Pair(totalCertainty, sumClass)
    }

    private fun calculateCertCGivenADiscrete(data: ArrayList<DataPoint>, attribute: Int) : Pair<Double, Double>{
        val probabilities = calculateAllProbabilities(data)
        var sum: Double
        var sum2 = 0.0
        var probability: Double
        var probabilityCAndA: Double
        var sumClass = 0.0

        val numValuesClass = domainsIndexToValue[classAttribute].size
        val numValuesGivenAtt = domainsIndexToValue[attribute].size

        for (j in 0 until numValuesGivenAtt) {
            if (domainsIndexToValue[attribute][j] == "?") continue
            probability = probabilities[attribute].prob[j]
            sum = 0.0
            for (i in 0 until numValuesClass) {
                probabilityCAndA = probabilities[attribute].probCAndA[j][i]
                sum += abs(probabilityCAndA - 1.0 * probability / numValuesClass)
            }
            sum2 += sum
        }
        //Calculate class certainty
        for (i in 0 until numValuesClass) {
            probability = probabilities[classAttribute].prob[i]
            sumClass += abs(probability - 1.0 * 1 / numValuesClass)
        }
        return Pair(sum2, sumClass)
    }

    private fun calculateProbCGivenADiscrete(data: ArrayList<DataPoint>, attribute: Int, attributeValue: Int): Pair<Int, Double>{
        val probabilities = calculateAllProbabilities(data)
        val numValuesClass = domainsIndexToValue[classAttribute].size
        var selectedClassValue = 0
        var greaterCondProbability = 0.0
        var condProbability: Double

        //val probability: Double = probabilities[attribute].prob[attributeValue]
        var probabilityCGivenA: Double

        for (i in 0 until numValuesClass) {
            probabilityCGivenA = probabilities[attribute].probCGivenA[attributeValue][i]
            condProbability = probabilityCGivenA
            if (condProbability > greaterCondProbability){
                greaterCondProbability = condProbability
                selectedClassValue = i
            }
        }
        return Pair(selectedClassValue, greaterCondProbability)
    }

    fun createCausalAnalysisReport(attributeName: String){
        val selectedAttributes = ArrayList<Int>()
        //First calculate class probabilities
        classProbabilities = calculateClassProbabilities()
        //Select ALL attributes
        for (i in 0 until numAttributes) {
            if (attributeTypes[i] == AttributeType.Ignore) continue
            selectedAttributes.add(i)
        }
        //In the following loop, the best attribute is located
        var certainty: Certainty
        for (selectedAtt in selectedAttributes) {
            if (classAttribute == selectedAtt) continue
            certainty = calculateCertainty(root.data, selectedAtt)
            //Insert into attributeImportance
            attributeImportance.add(Triple(selectedAtt, certainty.certainty, certainty.certaintyClass))
        }

        //If an attribute name is passed just do that attribute
        if (attributeName.isNotEmpty() && attributeName == "Class") {
            print("\n")
            print("Error: Attribute cannot be named \"Class\".")
            exitProcess(1)
        }
        else
        if (attributeName.isNotEmpty() && !attributeNames.contains(attributeName)){
            print("\n")
            print("Error: Attribute name not found.")
            exitProcess(1)
        }
        else
        if (attributeName.isNotEmpty() && attributeNames.contains(attributeName)){
            var selectedAtt: Triple<Int,Double,Double> = Triple(0,0.0,0.0)
            for (att in attributeImportance){
                if (attributeNames[att.first] == attributeName) {
                    selectedAtt = att
                    break
                }
            }
            attributeImportance.clear()
            attributeImportance.add(selectedAtt)
        }

        val sortedList: List<Triple<Int, Double, Double>> = if (criteria == Criteria.Certainty)
            attributeImportance.sortedWith(compareByDescending { it.second })
        else attributeImportance.sortedWith(compareBy { it.second })

        //this is needed to format console output
        var longestString: String?
        longestString = ""
        for ((i, element) in sortedList.withIndex()){
            if (i > 99) break
            val attName: String = attributeNames[element.first]
            if (longestString != null)
                if (attName.length > longestString.length) longestString = attName
        }
        if (longestString!!.length < 14) longestString = "--------------"

        //Stop the animation
        runAnimationAnalysis= false
        print("\r" + "[ ✓ ] " + "Analysis report created.")

        print("\n")
        print("\n")

        //Print console output
        if (this.criteria == Criteria.Certainty) {
            var fmt = "%1$10s %2$5s %3$1s %4$" + (longestString.length).toString() + "s%n"
            System.out.printf(fmt, "Importance", "Cause","", "Attribute Name")
            System.out.printf(fmt, "----------", "-----","", "--------------")
            for (i in sortedList.indices) {
                if (i > 99) break
                val rounded = String.format("%.2f", sortedList[i].second)
                val isCause = if (sortedList[i].second - sortedList[i].third > 0) "yes"
                else "no"
                val fillerSize = longestString.length - attributeNames[sortedList[i].first].length + 1
                val filler = String(CharArray(fillerSize)).replace('\u0000', '·')
                fmt = "%1$10s %2$5s %3$" + fillerSize.toString() + "s %4$" + attributeNames[sortedList[i].first].length.toString() + "s%n"
                System.out.printf(fmt, rounded, isCause, filler, attributeNames[sortedList[i].first])
            }
        }
        print("\n")
        print("Report of causal certainties")
        print("\n")
        print("----------------------------")
        print("\n")
        print("\n")

        //Variable declarations
        var probABelow: Double
        var probAAbove: Double
        var probCAndABelow: Double
        var probCAndAAbove: Double
        var condProbabilityBelow: Double
        var condProbabilityAbove: Double
        var condProbabilityBelowGreater: Double
        var condProbabilityAboveGreater: Double
        val numData = root.data.size
        val numValuesClass = domainsIndexToValue[classAttribute].size
        var selectedClassBelowValue: Int
        var selectedClassAboveValue: Int
        var threshold: Threshold?

        for (i in sortedList.indices){
            condProbabilityBelowGreater = 0.0
            condProbabilityAboveGreater = 0.0
            selectedClassBelowValue = 0
            selectedClassAboveValue = 0
            val isCause = sortedList[i].second - sortedList[i].third > 0
            if (isCause){
                print("[ Attribute: " + attributeNames[sortedList[i].first] + " ]")
                print("\n")
                print("\n")
                val list: ArrayList<String> = ArrayList()
                if (attributeTypes[sortedList[i].first] ==  AttributeType.Discrete){
                    for (j in 0 until domainsIndexToValue[sortedList[i].first].size){
                        val certCGivenA = calculateCertCGivenADiscrete(root.data, sortedList[i].first)
                        if (domainsIndexToValue[sortedList[i].first][j] == "?") continue

                        //Calculate causal certainties
                        val pair = calculateProbCGivenADiscrete(root.data, sortedList[i].first,j)
                        val selectedClassValue = pair.first
                        val probCGivenA = pair.second
                        val rounded = String.format("%.2f", probCGivenA)
                        val selectedClassValueName = domainsIndexToValue[classAttribute].getValue(selectedClassValue)
                        val probClass = classProbabilities.prob[selectedClassValue]
                        if (probCGivenA /certCGivenA.first > probClass / certCGivenA.second) {
                            //the elements of list will be printed out to console
                            list.add("    " + domainsIndexToValue[sortedList[i].first][j] + " --> " +
                                "$selectedClassValueName  ($rounded)"
                            )
                        }
                        else {
                            //the elements of list will be printed out to console
                            list.add("    " + domainsIndexToValue[sortedList[i].first][j] + " --> " +
                                "Non cause."
                            )
                        }
                    }
                    //print sorted elements of list
                    val comparator = NaturalOrderComparator()
                    val sortedList2: List<String> = list.sortedWith(comparator)

                    for (item in sortedList2){
                        print(item)

                        print("\n")
                        print("\n")
                    }
                }
                else
                if (attributeTypes[sortedList[i].first] ==  AttributeType.Continuous){
                    val cert = calculateCertainty(root.data,sortedList[i].first)
                    threshold = cert.thresholdObject
                    val certCGivenA = calculateCertCGivenAContinuous(root.data, threshold!!)
                    //Now calculate probabilities
                    probABelow = 1.0 * threshold.sumABelow / numData
                    probAAbove = 1.0 * threshold.sumAAbove / numData

                    //First below the threshold
                    for (c in 0 until numValuesClass) {
                        probCAndABelow = if (threshold.sumsClassesAndAttribute[c] != null) {
                            1.0 * threshold.sumsClassesAndAttribute[c]!!.below / numData
                        } else {
                            0.0
                        }

                        condProbabilityBelow = abs(1.0 * probCAndABelow/probABelow)
                        if (condProbabilityBelow > condProbabilityBelowGreater){
                            condProbabilityBelowGreater = condProbabilityBelow
                            selectedClassBelowValue = c
                        }
                    }
                    var roundedCertainty = String.format("%.2f", condProbabilityBelowGreater)
                    print("    <= " + String.format("%.2f", threshold.value))
                    print("  ")
                    var selectedClassValueName = domainsIndexToValue[classAttribute].getValue(selectedClassBelowValue)
                    var probClass = classProbabilities.prob[selectedClassBelowValue]
                    if (condProbabilityBelowGreater / certCGivenA.first > probClass / certCGivenA.second) {
                        print("--> $selectedClassValueName")
                        print("  ($roundedCertainty)")
                    }
                    else print("Non cause.")
                    print("\n")
                    print("\n")

                    //And then above the threshold
                    for (c in 0 until numValuesClass) {
                        probCAndAAbove = if (threshold.sumsClassesAndAttribute[c] != null) {
                            1.0 * threshold.sumsClassesAndAttribute[c]!!.above / numData
                        } else {
                            0.0
                        }

                        condProbabilityAbove = abs(1.0 * probCAndAAbove/probAAbove)
                        if (condProbabilityAbove > condProbabilityAboveGreater){
                            condProbabilityAboveGreater = condProbabilityAbove
                            selectedClassAboveValue = c
                        }
                    }
                    roundedCertainty = String.format("%.2f", condProbabilityAboveGreater)
                    print("    > " + String.format("%.2f", threshold.value))
                    print("  ")
                    selectedClassValueName = domainsIndexToValue[classAttribute].getValue(selectedClassAboveValue)
                    probClass = classProbabilities.prob[selectedClassAboveValue]
                    if (condProbabilityAboveGreater / certCGivenA.first > probClass / certCGivenA.second) {
                        print("--> $selectedClassValueName")
                        print("  ($roundedCertainty)")
                    }
                    else print("Non cause.")
                    print("\n")
                    print("\n")
                }
            }
        }
    }

    fun createDecisionTree() {
        val start = Instant.now()
        val selectedAttributes = ArrayList<Int>()
        //First calculate class probabilities
        classProbabilities = calculateClassProbabilities()
        //Select ALL attributes
        for (i in 0 until numAttributes) {
            if (attributeTypes[i] == AttributeType.Ignore) continue
            selectedAttributes.add(i)
        }
        decomposeNode(root, selectedAttributes, 0)
        var current: Thread?
        while (globalThreads.isNotEmpty()) {
            globalThreads.removeAll(Collections.singleton(null))
            current = globalThreads.last()
            if (current!=null && !current.isAlive) {
                globalThreads.removeLast()
            }
        }

        //Stop the animation
        runAnimationCalculating = false
        print("\r" + "[ ✓ ] " + "Decision tree created.")

        print("\n")
        print("\n")
        countNodes(root)
        print("Rules: $totalRules")
        print("\n")
        print("Nodes: $totalNodes")
        print("\n")

        val sortedList: List<Triple<Int, Double, Double>> = if (criteria == Criteria.Certainty)
            attributeImportance.sortedWith(compareByDescending { it.second })
        else attributeImportance.sortedWith(compareBy { it.second })

        //this is needed to format console output
        var longestString: String?
        longestString = ""
        for ((i, element) in sortedList.withIndex()){
            if (i > 99) break
            val attName: String = attributeNames[element.first]
            if (longestString != null)
                if (attName.length > longestString.length) longestString = attName
        }
        if (longestString!!.length < 14) longestString = "--------------"
        print("\n")
        //Print console output
        if (this.criteria == Criteria.Certainty) {
            var fmt = "%1$10s %2$5s %3$1s %4$" + (longestString.length).toString() + "s%n"
            System.out.printf(fmt, "Importance", "Cause","", "Attribute Name")
            System.out.printf(fmt, "----------", "-----","", "--------------")
            for (i in sortedList.indices) {
                if (i > 99) break
                val rounded = String.format("%.2f", sortedList[i].second)
                val isCause = if (sortedList[i].second - sortedList[i].third > 0) "yes"
                else "no"
                val fillerSize = longestString.length - attributeNames[sortedList[i].first].length + 1
                val filler = String(CharArray(fillerSize)).replace('\u0000', '·')
                fmt = "%1$10s %2$5s %3$" + fillerSize.toString() + "s %4$" + attributeNames[sortedList[i].first].length.toString() + "s%n"
                System.out.printf(fmt, rounded, isCause, filler, attributeNames[sortedList[i].first])
            }
        }
        else {
            val fmt = "%1$10s %2$" + (longestString.length + 10).toString() + "s%n"
            System.out.printf(fmt, "Importance", "Attribute Name")
            System.out.printf(fmt, "----------", "--------------")
            for (i in sortedList.indices) {
                if (i > 99) break
                val rounded = String.format("%.2f", sortedList[i].second)
                System.out.printf(fmt, rounded, attributeNames[sortedList[i].first])
            }
        }
        print("\n")
        testDecisionTree()
        print("\n")

        val finish = Instant.now()
        val timeElapsed = Duration.between(start, finish)
        val timeElapsedString = formatDuration(timeElapsed)
        print("Time: $timeElapsedString")
        print("\n")
    }

    fun createCrossValidation() {
        val start = Instant.now()
        if (testDataExists) {
            trainData.addAll(testData)
            root.data = trainData
            countClasses()
        }
        val chunkSize = root.data.size / 10
        val modulus = root.data.size % 10
        var counter = chunkSize
        var counterChunks = 0
        //Randomize the data
        root.data.shuffle()

        //Initialize chunks
        repeat(10) {
            crossValidationChunks.add(ArrayList())
        }

        //First check if there is a remainder
        if (modulus != 0) {
            var i = 0
            while (i < root.data.size - modulus) {
                if (i < counter) {
                    crossValidationChunks[counterChunks].add(root.data[i])
                } else {
                    counter += chunkSize
                    counterChunks++
                    i--
                }
                i++
            }
            counter = 0
            for (j in root.data.size - modulus until root.data.size) {
                crossValidationChunks[counter].add(root.data[j])
                counter++
            }
        } else {
            var i = 0
            while (i < root.data.size) {
                if (i < counter) {
                    crossValidationChunks[counterChunks].add(root.data[i])
                } else {
                    counter += chunkSize
                    counterChunks++
                    i--
                }
                i++
            }
        }
        val threads = ArrayList<Thread>()
        for (i in 0..9) {
            val newRoot = TreeNode()
            val trainData = ArrayList<DataPoint>()
            for (k in 0..9) {
                if (k != i) trainData.addAll(crossValidationChunks[k])
            }
            newRoot.data = trainData
            newRoot.frequencyClasses = getFrequencies(newRoot.data)
            rootsCrossValidation.add(newRoot)

            //Create the cross-validation in parallel
            val selectedAttributes = ArrayList<Int>()
            //Select ALL attributes
            for (j in 0 until numAttributes - 1) {
                if (attributeTypes[j] == AttributeType.Ignore) continue
                selectedAttributes.add(j)
            }
            val thread = Thread { decomposeNode(rootsCrossValidation[i], selectedAttributes, 0) }
            threads.add(thread)
            thread.start()
        }
        while (threads.isNotEmpty()) {
            if (!threads[threads.size - 1].isAlive) threads.removeAt(threads.size - 1)
        }
        //Stop the animation
        runAnimationCalculating = false

        print("\r" + "[ ✓ ] " + "10-fold cross-validation created with " + root.data.size + " cases.")

        print("\n")
        testCrossValidation()
        val finish = Instant.now()
        val timeElapsed = Duration.between(start, finish)
        val timeElapsedString = formatDuration(timeElapsed)
        print("\n")
        print("Time: $timeElapsedString")
        print("\n")
    }

    fun createCrossValidationRF() {
        val start = Instant.now()

        //Initialize array
        repeat(10) {
            cvRandomForests.add(ArrayList())
        }
        if (testDataExists) {
            trainData.addAll(testData)
            root.data = trainData
            countClasses()
        }
        val chunkSize = root.data.size / 10
        val modulus = root.data.size % 10
        var counter = chunkSize
        var counterChunks = 0
        //Randomize the data
        root.data.shuffle()

        //Initialize chunks
        repeat(10) {
            crossValidationChunks.add(ArrayList())
        }
        //First check if there is a remainder
        if (modulus != 0) {
            var i = 0
            while (i < root.data.size - modulus) {
                if (i < counter) {
                    crossValidationChunks[counterChunks].add(root.data[i])
                } else {
                    counter += chunkSize
                    counterChunks++
                    i--
                }
                i++
            }
            counter = 0
            for (j in root.data.size - modulus until root.data.size) {
                crossValidationChunks[counter].add(root.data[j])
                counter++
            }
        } else {
            var i = 0
            while (i < root.data.size) {
                if (i < counter) {
                    crossValidationChunks[counterChunks].add(root.data[i])
                } else {
                    counter += chunkSize
                    counterChunks++
                    i--
                }
                i++
            }
        }

        //Create the 10 Random Forests
        for (i in 0..9) {
            val trainData = ArrayList<DataPoint>()
            for (k in 0..9) {
                if (k != i) trainData.addAll(crossValidationChunks[k])
            }
            createRandomForest(trainData, cvRandomForests[i], true)
        }
        //Stop the animation
        runAnimationCalculating = false
        print("\r" + "[ ✓ ] " + "10-fold Random Forests cross-validation created with " + root.data.size + " cases.")

        print("\n")

        //Test the cross-validation
        testCrossValidationRF()
        val finish = Instant.now()
        val timeElapsed = Duration.between(start, finish)
        val timeElapsedString = formatDuration(timeElapsed)
        print("\n")
        print("Time: $timeElapsedString")
        print("\n")
    }

    fun createRandomForest(data: ArrayList<DataPoint>, roots: ArrayList<TreeNode>, cv: Boolean) {
        val start = Instant.now()
        var numberOfAttributes = 0
        //First calculate class probabilities
        classProbabilities = calculateClassProbabilities()
        //Second calculate importance certainties
        calculateImportanceCertainties()

        for (i in 0 until numAttributes - 1) {
            if (attributeTypes[i] != AttributeType.Ignore) numberOfAttributes++
        }
        val numAttributesForRandomForest = ln(numberOfAttributes + 1.toDouble()) / ln(2.0)
        val numAttributesForRandomForestInt = numAttributesForRandomForest.toInt()
        var randomAttribute: Int
        var selectedAttributes: ArrayList<Int>
        val threads = ArrayList<Thread>()
        val rand = Random(seed)
        val freqClasses  = getFrequencies(data)
        for (i in 0 until numberOfTrees) {
            selectedAttributes = ArrayList()
            while (selectedAttributes.size < numAttributesForRandomForestInt) {
                randomAttribute = rand.nextInt(numAttributes - 1)
                if (!selectedAttributes.contains(randomAttribute) && attributeTypes[randomAttribute] != AttributeType.Ignore) selectedAttributes.add(randomAttribute)
            }
            val cloneRoot = TreeNode()
            cloneRoot.data = data
            cloneRoot.frequencyClasses = freqClasses
            roots.add(cloneRoot)

            //Create the Random Forest in parallel
            val selectedAttributes2 = selectedAttributes
            val thread = Thread { decomposeNode(roots[i], selectedAttributes2, seed + 1 + i) }
            threads.add(thread)
            thread.start()
        }
        while (threads.isNotEmpty()) {
            if (!threads[threads.size - 1].isAlive) threads.removeAt(threads.size - 1)
        }
        if (!cv) {
            //Stop the animation
            runAnimationCalculating = false
            print("\r" + "[ ✓ ] " + "Random Forest of " + roots.size + " trees created.")

            print("\n")
            val sortedList: List<Triple<Int, Double, Double>> = if (criteria == Criteria.Certainty)
                attributeImportance.sortedWith(compareByDescending { it.second })
            else attributeImportance.sortedWith(compareBy { it.second })

            //this is needed to format console output
            var longestString: String?
            longestString = ""
            for ((i, element) in sortedList.withIndex()){
                if (i > 99) break
                val attName: String = attributeNames[element.first]
                if (longestString != null)
                    if (attName.length > longestString.length) longestString = attName
            }
            if (longestString!!.length < 14) longestString = "--------------"
            print("\n")
            //Print console output...show attribute importance
            if (this.criteria == Criteria.Certainty) {
                var fmt = "%1$10s %2$5s %3$1s %4$" + (longestString.length).toString() + "s%n"
                System.out.printf(fmt, "Importance", "Cause","", "Attribute Name")
                System.out.printf(fmt, "----------", "-----","", "--------------")
                for (i in sortedList.indices) {
                    if (i > 99) break
                    val rounded = String.format("%.2f", sortedList[i].second)
                    val isCause = if (sortedList[i].second - sortedList[i].third > 0) "yes"
                    else "no"
                    val fillerSize = longestString.length - attributeNames[sortedList[i].first].length + 1
                    val filler = String(CharArray(fillerSize)).replace('\u0000', '·')
                    fmt = "%1$10s %2$5s %3$" + fillerSize.toString() + "s %4$" + attributeNames[sortedList[i].first].length.toString() + "s%n"
                    System.out.printf(fmt, rounded, isCause, filler, attributeNames[sortedList[i].first])
                }
            }
            else {
                val fmt = "%1$10s %2$" + (longestString.length + 10).toString() + "s%n"
                System.out.printf(fmt, "Importance", "Attribute Name")
                System.out.printf(fmt, "----------", "--------------")
                for (i in sortedList.indices) {
                    if (i > 99) break
                    val rounded = String.format("%.2f", sortedList[i].second)
                    System.out.printf(fmt, rounded, attributeNames[sortedList[i].first])
                }
            }
            print("\n")
            testRandomForest()
            val finish = Instant.now()
            val timeElapsed = Duration.between(start, finish)
            val timeElapsedString = formatDuration(timeElapsed)
            print("\n")
            print("\n")
            print("Time: $timeElapsedString")
            print("\n")
        }
    }

    private fun testExamplePoint(example: DataPoint, node: TreeNode): TreeNode {
        var nodeLocal = node
        val attributeValue: Int
        val attributeRealValue: Double
        val splitAttribute: Int = nodeLocal.decompositionAttribute
        attributeValue = example.attributes[splitAttribute]
        if (nodeLocal.children.isEmpty()) return nodeLocal

        //Check if attribute is discrete
        if (attributeTypes[splitAttribute] == AttributeType.Discrete) {
            for (i in nodeLocal.children.indices) {
                if (nodeLocal.children[i].decompositionValue == attributeValue) {
                    nodeLocal = nodeLocal.children[i]
                    break
                }
            }
        }
        //Check if attribute is continuous
        if (attributeTypes[splitAttribute] == AttributeType.Continuous) {
            attributeRealValue = (domainsIndexToValue[splitAttribute][example.attributes[splitAttribute]])!!.toDouble()
            nodeLocal = if (attributeRealValue <= nodeLocal.children[0].thresholdContinuous) {
                nodeLocal.children[0]
            } else nodeLocal.children[1]
        }
        return testExamplePoint(example, nodeLocal)
    }

    private fun testExample(example: DataPoint, train: Boolean): Boolean {
        val node: TreeNode = testExamplePoint(example, root)
        if (node.data.isEmpty()) {
            return if (example.attributes[classAttribute] == getMostCommonClass(node.parent)){
                true
            } else {
                val currentClass = getMostCommonClass(node.parent)
                when (train){
                    true -> {
                        falsePositivesTrain[currentClass]++
                        falseNegativesTrain[example.attributes[classAttribute]]++
                    }
                    else ->{
                        falsePositivesTest[currentClass]++
                        falseNegativesTest[example.attributes[classAttribute]]++
                    }
                }
                false
            }
        }
        else {
            return if(example.attributes[classAttribute] == getMostCommonClass(node)){
                true
            } else {
                val currentClass = getMostCommonClass(node)
                when (train){
                    true -> {
                        falsePositivesTrain[currentClass]++
                        falseNegativesTrain[example.attributes[classAttribute]]++
                    }
                    else ->{
                        falsePositivesTest[currentClass]++
                        falseNegativesTest[example.attributes[classAttribute]]++
                    }
                }
                false
            }
        }
    }

    private fun testExampleCV(example: DataPoint, tree: TreeNode): Boolean {
        val node: TreeNode = testExamplePoint(example, tree)
        if (node.data.isEmpty()) {
            return if (example.attributes[classAttribute] == getMostCommonClass(node.parent)){
                true
            } else {
                val currentClass = getMostCommonClass(node.parent)
                falsePositivesTest[currentClass]++
                falseNegativesTest[example.attributes[classAttribute]]++
                false
            }
        }
        else {
            return if(example.attributes[classAttribute] == getMostCommonClass(node)){
                true
            } else {
                val currentClass = getMostCommonClass(node)
                falsePositivesTest[currentClass]++
                falseNegativesTest[example.attributes[classAttribute]]++
                false
            }
        }
    }

    private fun testExampleRF(example: DataPoint, roots: ArrayList<TreeNode>, train: Boolean): Boolean {
        var node: TreeNode
        var isTrue = 0
        var isFalse = 0
        var currentClass: Int
        val results = ArrayList<Boolean>()
        val classifiedAs = IntArray(domainsIndexToValue[numAttributes - 1].size)

        for (treeNode in roots) {
            node = testExamplePoint(example, treeNode)
            if (node.data.isEmpty()) {
                currentClass = getMostCommonClass(node.parent)
                if (example.attributes[classAttribute] == currentClass) results.add(true)
                else {
                    classifiedAs[currentClass]++
                    results.add(false)
                }
            } else {
                currentClass = getMostCommonClass(node)
                if (example.attributes[classAttribute] == currentClass) results.add(true)
                else {
                    classifiedAs[currentClass]++
                    results.add(false)
                }
            }
        }

        //Voting now
        for (result in results) {
            if (result) isTrue++ else isFalse++
        }
        if (isTrue > isFalse) {
            return true
        }
        else {
            var count = 0
            var maxCountClass = 0
            for (i in classifiedAs.indices){
                if (classifiedAs[i] > count) {
                    count = classifiedAs[i]
                    maxCountClass = i
                }
            }
            when (train) {
                true -> {
                    falsePositivesTrain[maxCountClass]++
                    falseNegativesTrain[example.attributes[classAttribute]]++
                }
                else -> {
                    falsePositivesTest[maxCountClass]++
                    falseNegativesTest[example.attributes[classAttribute]]++
                }
            }
            return false
        }
    }

    private fun testDecisionTree() {
        var trainErrors = 0
        var trainCorrects = 0

        //this is needed to format console output
        var longestString: String?
        longestString = ""
        for (i in falsePositivesTrain.indices){
            val classValue: String = domainsIndexToValue[numAttributes - 1][i] as String
            if (longestString != null)
                if (classValue.length > longestString.length) longestString = classValue
        }

        if (longestString!!.length < 5) longestString = "-----"

        for (point in trainData) {
            if (testExample(point, true)) trainCorrects++ else trainErrors++
        }
        print("\n")
        print("[==== TRAIN DATA ====] ")
        print("\n")
        print("\n")

        var rounded1 = (1.0 * trainErrors * 100 / trainData.size * 10).roundToInt() / 10.0
        var format : String = "%" + (if (trainCorrects.toString().length - trainErrors.toString().length > 0) trainCorrects.toString().length
        else trainErrors.toString().length) + "d"
        print( "Correct guesses:   ")
        print(String.format(format, trainCorrects))
        print("\n")
        print( "Incorrect guesses: ")
        print(String.format(format, trainErrors))
        print( " ($rounded1%)")

        print("\n")
        print("\n")

        var fmt = "%1$10s %2$10s %3$10s %4$1s %5$" + (longestString.length).toString() + "s%n"
        System.out.printf(fmt, "# Of Cases", "False Pos", "False Neg", "", "Class")
        System.out.printf(fmt, "----------", "---------", "---------", "", "-----")
        for (value in domainsValueToIndex[numAttributes - 1].keys){
            val i = domainsValueToIndex[numAttributes - 1][value] as Int
            val fillerSize = longestString.length - (domainsIndexToValue[numAttributes - 1][i] as String).length + 1
            val filler = String(CharArray(fillerSize)).replace('\u0000', '·')
            fmt = "%1$10s %2$10s %3$10s %4$" + fillerSize.toString() + "s %5$" + ((domainsIndexToValue[numAttributes - 1][i] as String).length).toString() + "s%n"
            val cases = if (save) (classNoOfCasesTrain[i] + classNoOfCasesTest[i]).toString()
            else (classNoOfCasesTrain[i]).toString()
            System.out.printf(fmt, cases, falsePositivesTrain[i].toString(), falseNegativesTrain[i].toString(), filler, domainsIndexToValue[numAttributes - 1][i] as String)
        }

        if (testData.isNotEmpty()) {
            var testErrors = 0
            var testCorrects = 0
            for (point in testData) {
                if (testExample(point, false)) testCorrects++ else testErrors++
            }

            print("\n")
            print("[==== TEST DATA ====] ")
            print("\n")
            print("\n")

            rounded1 = (1.0 * testErrors * 100 / testData.size * 10).roundToInt() / 10.0
            format = "%" + (if (testCorrects.toString().length - testErrors.toString().length > 0) testCorrects.toString().length
            else testErrors.toString().length) + "d"
            print( "Correct guesses:   ")
            print(String.format(format, testCorrects))
            print("\n")
            print( "Incorrect guesses: ")
            print(String.format(format, testErrors))
            print( " ($rounded1%)")

            print("\n")
            print("\n")

            fmt = "%1$10s %2$10s %3$10s %4$1s %5$" + (longestString.length).toString() + "s%n"
            System.out.printf(fmt, "# Of Cases", "False Pos", "False Neg", "", "Class")
            System.out.printf(fmt, "----------", "---------", "---------", "", "-----")
            for (value in domainsValueToIndex[numAttributes - 1].keys){
                val i = domainsValueToIndex[numAttributes - 1][value] as Int
                val fillerSize = longestString.length - (domainsIndexToValue[numAttributes - 1][i] as String).length + 1
                val filler = String(CharArray(fillerSize)).replace('\u0000', '·')
                fmt = "%1$10s %2$10s %3$10s %4$" + fillerSize.toString() + "s %5$" + ((domainsIndexToValue[numAttributes - 1][i] as String).length).toString() + "s%n"
                System.out.printf(fmt, classNoOfCasesTest[i].toString(), falsePositivesTest[i].toString(), falseNegativesTest[i].toString(), filler, domainsIndexToValue[numAttributes - 1][i] as String)
            }
        }
    }

    private fun testCrossValidation() {
        var testErrors: Int
        val meanErrors: Double
        var percentageErrors = 0.0
        val errorsFoldK = DoubleArray(10)
        val roundedErrors = ArrayList<Double>()
        for (i in 0..9) {
            testErrors = 0
            val currentTree = rootsCrossValidation[i]
            val currentTest = crossValidationChunks[i]
            for (point in currentTest) {
                if (!testExampleCV(point, currentTree)) testErrors++
            }
            percentageErrors += 1.0 * testErrors / currentTest.size * 100
            val rounded1 = (1.0 * testErrors / currentTest.size * 100 * 10).roundToInt() / 10.0
            roundedErrors.add(rounded1)
            //Save k errors for SE
            errorsFoldK[i] = 1.0 * testErrors / currentTest.size * 100
        }

        //Print console output
        print("\n")
        var fmt = "%1$4s %2$10s%n"
        System.out.printf(fmt, "Fold", "Errors")
        System.out.printf(fmt, "-----", "-------")
        for (i in 0 until 10) {
            System.out.printf(fmt, (i + 1).toString(), roundedErrors[i].toString() + "%")
        }

        //Print mean errors
        meanErrors = percentageErrors / 10
        val rounded1 = (meanErrors * 10).roundToInt() / 10.0
        System.out.printf(fmt, " ", " ")
        System.out.printf(fmt, "Mean", "$rounded1%")

        //Calculate average
        var meanFolds = 0.0
        for (i in 0..9) {
            meanFolds += errorsFoldK[i]
        }
        meanFolds = 1.0 * meanFolds / 10

        //Calculate SE (Standard Errors)
        var sumMeanSE = 0.0
        for (i in 0..9) {
            sumMeanSE += (1.0 * errorsFoldK[i] - meanFolds) * (1.0 * errorsFoldK[i] - meanFolds)
        }
        sumMeanSE = sqrt(sumMeanSE / 10)
        val se = sumMeanSE / sqrt(10.0)
        val roundedSE = (se * 10).roundToInt() / 10.0
        System.out.printf(fmt, "SE", "$roundedSE%")

        //Now print the False Positives/Negatives
        //this is needed to format console output
        var longestString: String?
        longestString = ""
        for (i in falsePositivesTrain.indices){
            val classValue: String = domainsIndexToValue[numAttributes - 1][i] as String
            if (longestString != null)
                if (classValue.length > longestString.length) longestString = classValue
        }
        if (longestString!!.length < 5) longestString = "-----"

        print("\n")
        print("\n")
        fmt = "%1$10s %2$10s %3$10s %4$1s %5$" + (longestString.length).toString() + "s%n"
        System.out.printf(fmt, "# Of Cases", "False Pos", "False Neg", "", "Class")
        System.out.printf(fmt, "----------", "---------", "---------", "", "-----")
        for (value in domainsValueToIndex[numAttributes - 1].keys){
            val i = domainsValueToIndex[numAttributes - 1][value] as Int
            val fillerSize = longestString.length - (domainsIndexToValue[numAttributes - 1][i] as String).length + 1
            val filler = String(CharArray(fillerSize)).replace('\u0000', '·')
            fmt = "%1$10s %2$10s %3$10s %4$" + fillerSize.toString() + "s %5$" + ((domainsIndexToValue[numAttributes - 1][i] as String).length).toString() + "s%n"
            System.out.printf(fmt, classNoOfCasesTrain[i].toString(), falsePositivesTest[i].toString(), falseNegativesTest[i].toString(), filler, domainsIndexToValue[numAttributes - 1][i] as String)
        }
    }

    private fun testCrossValidationRF() {
        var sum = 0.0
        val errorsFoldK = DoubleArray(10)
        var current: Double
        val roundedErrors = ArrayList<Double>()
        //For each Random Forest
        for (i in 0..9) {
            val currentForest = cvRandomForests[i]
            val currentTestData = crossValidationChunks[i]
            current = testRandomForest(currentTestData, currentForest)
            roundedErrors.add(current)
            sum += current
            //Save k errors for SE
            errorsFoldK[i] = current
        }

        //Print console output
        print("\n")
        var fmt = "%1$4s %2$10s%n"
        System.out.printf(fmt, "Fold", "Errors")
        System.out.printf(fmt, "-----", "-------")
        for (i in 0 until 10) {
            System.out.printf(fmt, (i + 1).toString(), roundedErrors[i].toString() + "%")
        }

        val meanErrors = sum / 10
        val rounded1 = (meanErrors * 10).roundToInt() / 10.0
        System.out.printf(fmt, " ", " ")
        System.out.printf(fmt, "Mean", "$rounded1%")

        //Calculate SE (Standard Errors)
        var sumMeanSE = 0.0
        for (i in 0..9) {
            sumMeanSE += (1.0 * errorsFoldK[i] - meanErrors) * (1.0 * errorsFoldK[i] - meanErrors)
        }
        sumMeanSE = sqrt(sumMeanSE / 10)
        val se = sumMeanSE / sqrt(10.0)
        val roundedSE = (se * 10).roundToInt() / 10.0
        System.out.printf(fmt, "SE", "$roundedSE%")

        //Now print the False Positives/Negatives
        //this is needed to format console output
        var longestString: String?
        longestString = ""
        for (i in falsePositivesTrain.indices){
            val classValue: String = domainsIndexToValue[numAttributes - 1][i] as String
            if (longestString != null)
                if (classValue.length > longestString.length) longestString = classValue
        }
        if (longestString!!.length < 5) longestString = "-----"

        print("\n")
        print("\n")
        fmt = "%1$10s %2$10s %3$10s %4$1s %5$" + (longestString.length).toString() + "s%n"
        System.out.printf(fmt, "# Of Cases", "False Pos", "False Neg", "", "Class")
        System.out.printf(fmt, "----------", "---------", "---------", "", "-----")
        for (value in domainsValueToIndex[numAttributes - 1].keys){
            val i = domainsValueToIndex[numAttributes - 1][value] as Int
            val fillerSize = longestString.length - (domainsIndexToValue[numAttributes - 1][i] as String).length + 1
            val filler = String(CharArray(fillerSize)).replace('\u0000', '·')
            fmt = "%1$10s %2$10s %3$10s %4$" + fillerSize.toString() + "s %5$" + ((domainsIndexToValue[numAttributes - 1][i] as String).length).toString() + "s%n"
            System.out.printf(fmt, classNoOfCasesTrain[i].toString(), falsePositivesTest[i].toString(), falseNegativesTest[i].toString(), filler, domainsIndexToValue[numAttributes - 1][i] as String)
        }
    }

    //This overload method is intended to be used when Random Forest cross-validation is selected.
    private fun testRandomForest(testD: ArrayList<DataPoint>, roots: ArrayList<TreeNode>): Double {
        var testErrors = 0
        val testSize = testD.size
        for (point in testD) {
            if (!testExampleRF(point, roots, false)) testErrors++
        }
        return (1.0 * testErrors * 100 / testSize * 10).roundToInt() / 10.0
    }

    private fun testRandomForest() {
        //this is needed to format console output
        var longestString: String?
        longestString = ""
        for (i in falsePositivesTrain.indices){
            val classValue: String = domainsIndexToValue[numAttributes - 1][i] as String
            if (longestString != null)
                if (classValue.length > longestString.length) longestString = classValue
        }

        if (longestString!!.length < 5) longestString = "-----"

        var trainErrors = 0
        var trainCorrects = 0
        for (point in trainData) {
            if (testExampleRF(point, rootsRandomForest, true)) trainCorrects++ else trainErrors++
        }
        print("[==== TRAIN DATA ====] ")
        print("\n")
        print("\n")

        var rounded1 = (1.0 * trainErrors * 100 / trainData.size * 10).roundToInt() / 10.0
        var format : String = "%" + (if (trainCorrects.toString().length - trainErrors.toString().length > 0) trainCorrects.toString().length
        else trainErrors.toString().length) + "d"
        print( "Correct guesses:   ")
        print(String.format(format, trainCorrects))
        print("\n")
        print( "Incorrect guesses: ")
        print(String.format(format, trainErrors))
        print( " ($rounded1%)")

        print("\n")
        print("\n")

        var fmt = "%1$10s %2$10s %3$10s %4$1s %5$" + (longestString.length).toString() + "s%n"
        System.out.printf(fmt, "# Of Cases", "False Pos", "False Neg", "", "Class")
        System.out.printf(fmt, "----------", "---------", "---------", "", "-----")
        for (value in domainsValueToIndex[numAttributes - 1].keys){
            val i = domainsValueToIndex[numAttributes - 1][value] as Int
            val fillerSize = longestString.length - (domainsIndexToValue[numAttributes - 1][i] as String).length + 1
            val filler = String(CharArray(fillerSize)).replace('\u0000', '·')
            fmt = "%1$10s %2$10s %3$10s %4$" + fillerSize.toString() + "s %5$" + ((domainsIndexToValue[numAttributes - 1][i] as String).length).toString() + "s%n"
            val cases = if (save) (classNoOfCasesTrain[i] + classNoOfCasesTest[i]).toString()
            else (classNoOfCasesTrain[i]).toString()
            System.out.printf(fmt, cases, falsePositivesTrain[i].toString(), falseNegativesTrain[i].toString(),filler, domainsIndexToValue[numAttributes - 1][i] as String)
        }

        if (testData.isNotEmpty()) {
            var testErrors = 0
            var testCorrects = 0
            for (point in testData) {
                if (testExampleRF(point, rootsRandomForest, false)) testCorrects++ else testErrors++
            }
            print("\n")
            print("[==== TEST DATA ====] ")
            print("\n")
            print("\n")

            rounded1 = (1.0 * testErrors * 100 / testData.size * 10).roundToInt() / 10.0
            format = "%" + (if (testCorrects.toString().length - testErrors.toString().length > 0) testCorrects.toString().length
            else testErrors.toString().length) + "d"
            print( "Correct guesses:   ")
            print(String.format(format, testCorrects))
            print("\n")
            print( "Incorrect guesses: ")
            print(String.format(format, testErrors))
            print( " ($rounded1%)")

            print("\n")
            print("\n")

            fmt = "%1$10s %2$10s %3$10s %4$1s %5$" + (longestString.length).toString() + "s%n"
            System.out.printf(fmt, "# Of Cases", "False Pos", "False Neg", "", "Class")
            System.out.printf(fmt, "----------", "---------", "---------", "", "-----")
            for (value in domainsValueToIndex[numAttributes - 1].keys){
                val i = domainsValueToIndex[numAttributes - 1][value] as Int
                val fillerSize = longestString.length - (domainsIndexToValue[numAttributes - 1][i] as String).length + 1
                val filler = String(CharArray(fillerSize)).replace('\u0000', '·')
                fmt = "%1$10s %2$10s %3$10s %4$" + fillerSize.toString() + "s %5$" + ((domainsIndexToValue[numAttributes - 1][i] as String).length).toString() + "s%n"
                System.out.printf(fmt, classNoOfCasesTest[i], falsePositivesTest[i].toString(), falseNegativesTest[i].toString(), filler, domainsIndexToValue[numAttributes - 1][i] as String)
            }
        }
    }

    fun queryTree(file: String) {
        var fileLocal = file
        if (!fileLocal.endsWith(".tree")) fileLocal += ".tree"
        val inputFile = File(fileLocal)
        val id3: Cid3?
        if (inputFile.exists()) {
            val `in` = Scanner(System.`in`)
            id3 = deserializeFile(fileLocal)
            print("\n")
            print("Tree file deserialized.")
            print("\n")
            //Show version and time
            print(id3.createdWith)
            print("\n")

            //First display data statistics
            val sortedList: List<Triple<Int, Double, Double>> = if (id3.criteria == Criteria.Certainty)
                id3.attributeImportance.sortedWith(compareByDescending { it.second })
            else id3.attributeImportance.sortedWith(compareBy { it.second })

            //this is needed to format console output
            var longestString: String?
            longestString = ""
            for ((i, element) in sortedList.withIndex()){
                if (i > 99) break
                val attName: String = id3.attributeNames[element.first]
                if (longestString != null)
                    if (attName.length > longestString.length) longestString = attName
            }
            if (longestString!!.length < 14) longestString = "--------------"
            print("\n")
            //Print console output
            if (id3.criteria == Criteria.Certainty) {
                var fmt = "%1$10s %2$5s %3$1s %4$" + (longestString.length).toString() + "s%n"
                System.out.printf(fmt, "Importance", "Cause","", "Attribute Name")
                System.out.printf(fmt, "----------", "-----","", "--------------")
                for (i in sortedList.indices) {
                    if (i > 99) break
                    val rounded = String.format("%.2f", sortedList[i].second)
                    val isCause = if (sortedList[i].second - sortedList[i].third > 0) "yes"
                    else "no"
                    val fillerSize = longestString.length - id3.attributeNames[sortedList[i].first].length + 1
                    val filler = String(CharArray(fillerSize)).replace('\u0000', '·')
                    fmt = "%1$10s %2$5s %3$" + fillerSize.toString() + "s %4$" + id3.attributeNames[sortedList[i].first].length.toString() + "s%n"
                    System.out.printf(fmt, rounded, isCause, filler, id3.attributeNames[sortedList[i].first])
                }
            }
            else {
                val fmt = "%1$10s %2$" + (longestString.length + 10).toString() + "s%n"
                System.out.printf(fmt, "Importance", "Attribute Name")
                System.out.printf(fmt, "----------", "--------------")
                for (i in sortedList.indices) {
                    if (i > 99) break
                    val rounded = String.format("%.2f", sortedList[i].second)
                    System.out.printf(fmt, rounded, id3.attributeNames[sortedList[i].first])
                }
            }
            print("\n")

            var currentNode = id3.root
            var attributeValue = 0
            while (currentNode.children.isNotEmpty()) {
                //If attribute is discrete, show all possible values for it
                if (id3.attributeTypes[currentNode.decompositionAttribute] == AttributeType.Discrete) {
                    print("Please enter attribute: " + id3.attributeNames[currentNode.decompositionAttribute])
                    print("\n")
                    print("(possible values are: ")
                    var values = StringBuilder()
                    val valuesArray = ArrayList<String>()
                    for (i in 0 until id3.domainsIndexToValue[currentNode.decompositionAttribute].size) {
                        if (id3.domainsIndexToValue[currentNode.decompositionAttribute][i] == "?") continue
                        valuesArray.add(id3.domainsIndexToValue[currentNode.decompositionAttribute][i] as String)
                    }
                    val comparator = NaturalOrderComparator()
                    val valuesList :List<String> = valuesArray.sortedWith(comparator).toList()
                    for (value in valuesList) {
                        values.append(value)
                        values.append(", ")
                    }
                    values = StringBuilder("?, " + values.substring(0, values.length - 2))
                    print("$values)")
                    print("\n")
                    while (true) {
                        try {
                            val s = `in`.nextLine()
                            if (s == "?") {
                                attributeValue = id3.mostCommonValues[currentNode.decompositionAttribute]
                                break
                            } else attributeValue = id3.domainsValueToIndex[currentNode.decompositionAttribute][s]!!
                            break
                        } catch (_: Exception) {
                            println("Please enter a valid value:")
                        }
                    }
                }
                for (i in currentNode.children.indices) {
                    //Check if attribute is continuous
                    if (id3.attributeTypes[currentNode.decompositionAttribute] == AttributeType.Continuous) {
                        if (currentNode.children[i].decompositionValueContinuousSymbol == "<=") {
                            print("Is attribute: " + id3.attributeNames[currentNode.decompositionAttribute] + " <= " + currentNode.children[i].thresholdContinuous + " ? (y/n/?)")
                            print("\n")
                            while (true) {
                                val s = `in`.nextLine()
                                if (s == "y" || s == "Y" || s == "yes" || s == "Yes" || s == "YES" || s == "n" || s == "N" || s == "no" || s == "No" || s == "NO" || s == "?") {
                                    if (s == "y" || s == "Y" || s == "yes" || s == "Yes" || s == "YES") {
                                        currentNode = currentNode.children[i]
                                        break
                                    } else if (s == "?") {
                                        val mean = id3.meanValues[currentNode.decompositionAttribute]
                                        currentNode = if (mean <= currentNode.children[i].thresholdContinuous) {
                                            currentNode.children[i]
                                        } else {
                                            currentNode.children[i + 1]
                                        }
                                        break
                                    } else {
                                        currentNode = currentNode.children[i + 1]
                                        break
                                    }
                                }
                                else {
                                    print("Please enter a valid value:")
                                    print("\n")
                                }
                            }
                            break
                        }
                    } else if (currentNode.children[i].decompositionValue == attributeValue) {
                        currentNode = currentNode.children[i]
                        break
                    }
                }
            }
            //Check if the node is empty, if so, return its parent most frequent class.
            var isEmpty = true
            for (i in currentNode.frequencyClasses.indices) {
                if (currentNode.frequencyClasses[i] != 0) {
                    isEmpty = false
                    break
                }
            }
            val mostCommon: Int = if (isEmpty) id3.getMostCommonClass(currentNode.parent) else id3.getMostCommonClass(currentNode)
            val mostCommonStr = id3.domainsIndexToValue[id3.classAttribute][mostCommon]
            //Print class attribute value
            println("Class attribute value is: $mostCommonStr")
        } else println("The file doesn't exist.")
    }

    fun queryTreeOutput(treeFile: String, casesFile: String) {
        var treeFileLocal = treeFile
        var casesFileLocal = casesFile
        if (!treeFileLocal.endsWith(".tree")) treeFileLocal += ".tree"
        val fileOutStr: String = if (!casesFileLocal.endsWith(".cases")) "$casesFileLocal.tmp" else {
            casesFileLocal.substring(0, casesFileLocal.length - 5) + "tmp"
        }
        val inputTreeFile = File(treeFileLocal)
        val id3: Cid3?
        val fileOut: FileWriter?
        try {
            fileOut = FileWriter(fileOutStr, false)
        }
        catch (_: Exception) {
            System.err.println("Error creating temporal file.")
            exitProcess(1)
        }
        val fileBuf = BufferedWriter(fileOut)
        val printOut = PrintWriter(fileBuf)
        if (inputTreeFile.exists()) {
            id3 = deserializeFile(treeFileLocal)
            print("\n")
            print("Tree file deserialized.")
            print("\n")
            val inCases: FileInputStream?
            try {
                if (!casesFileLocal.endsWith(".cases")) casesFileLocal += ".cases"
                val inputFile = File(casesFileLocal)
                inCases = FileInputStream(inputFile)
            }
            catch (_: Exception) {
                System.err.println("Unable to open cases file.")
                exitProcess(1)
            }
            val bin = BufferedReader(InputStreamReader(inCases))
            var input: String?
            var tokenizer: StringTokenizer
            try {
                input = bin.readLine()
            }
            catch (_: Exception) {
                System.err.println("Unable to read line:")
                exitProcess(1)
            }
            while (input != null) {
                if (input.trim { it <= ' ' } == "") continue
                if (input.startsWith("//")) continue
                tokenizer = StringTokenizer(input, ",")
                val numTokens = tokenizer.countTokens()
                if (numTokens != id3.numAttributes - 1) {
                    System.err.println("Expecting " + (id3.numAttributes - 1) + " attributes")
                    exitProcess(1)
                }
                val point = DataPoint(id3.numAttributes)
                var next: String
                for (i in 0 until id3.numAttributes - 1) {
                    next = tokenizer.nextToken().trim { it <= ' ' }
                    if (id3.attributeTypes[i] == AttributeType.Continuous) {
                        if (next == "?" || next == "NaN") {
                            val value: Double = id3.meanValues[i]
                            point.attributes[i] = id3.getSymbolValue(i, value.toString())
                        } else {
                            try {
                                point.attributes[i] = id3.getSymbolValue(i, next)
                            } catch (_: Exception) {
                                System.err.println("Error reading continuous value in train data.")
                                exitProcess(1)
                            }
                        }
                    } else if (id3.attributeTypes[i] == AttributeType.Discrete) {
                        if (next == "?" || next == "NaN") {
                            point.attributes[i] = id3.mostCommonValues[i]
                        } else point.attributes[i] = id3.getSymbolValue(i, next)
                    }
                }
                //Test the example point
                val node: TreeNode = id3.testExamplePoint(point, id3.root)
                var isEmpty = true
                //Check if the node is empty, if so, return its parent most frequent class.
                for (j in node.frequencyClasses.indices) {
                    if (node.frequencyClasses[j] != 0) {
                        isEmpty = false
                        break
                    }
                }
                //If node is empty
                val caseClass: Int = if (isEmpty) id3.getMostCommonClass(node.parent) else id3.getMostCommonClass(node)

                //Print line to output tmp file
                val classValue = id3.domainsIndexToValue[id3.classAttribute][caseClass]
                val line = "$input,$classValue"
                printOut.write(line)
                printOut.println()

                //continue the loop
                try {
                    input = bin.readLine()
                }
                catch (_: Exception) {
                    System.err.println("Unable to read line.")
                    exitProcess(1)
                }
            }
            printOut.close()
            print("\n")
            print("Results saved to tmp file.")
            print("\n")
        } else println("The tree file doesn't exist.")
    }

    fun queryRandomForestOutput(rfFile: String, casesFile: String) {
        var rfFileLocal = rfFile
        var casesFileLocal = casesFile
        if (!rfFileLocal.endsWith(".forest")) rfFileLocal += ".forest"
        val fileOutStr: String = if (!casesFileLocal.endsWith(".cases")) "$casesFileLocal.tmp" else {
            casesFileLocal.substring(0, casesFileLocal.length - 5) + "tmp"
        }
        val inputForestFile = File(rfFileLocal)
        val id3: Cid3?
        val fileOut: FileWriter?
        try {
            fileOut = FileWriter(fileOutStr, false)
        } catch (_: Exception) {
            System.err.println("Error creating temporal file.")
            exitProcess(1)
        }
        val fileBuf = BufferedWriter(fileOut)
        val printOut = PrintWriter(fileBuf)
        if (inputForestFile.exists()) {
            id3 = deserializeFile(rfFileLocal)
            print("\n")
            print("Forest file deserialized.")
            print("\n")
            val inCases: FileInputStream?
            try {
                if (!casesFileLocal.endsWith(".cases")) casesFileLocal += ".cases"
                val inputFile = File(casesFileLocal)
                inCases = FileInputStream(inputFile)
            } catch (_: Exception) {
                System.err.println("Unable to open cases file.")
                exitProcess(1)
            }
            val bin = BufferedReader(InputStreamReader(inCases))
            var input: String?
            var tokenizer: StringTokenizer
            try {
                input = bin.readLine()
            }
            catch (_: Exception) {
                System.err.println("Unable to read line:")
                exitProcess(1)
            }
            while (input != null) {
                if (input.trim { it <= ' ' } == "") continue
                if (input.startsWith("//")) continue
                tokenizer = StringTokenizer(input, ",")
                val numTokens = tokenizer.countTokens()
                if (numTokens != id3.numAttributes - 1) {
                    System.err.println("Expecting " + (id3.numAttributes - 1) + " attributes")
                    exitProcess(1)
                }
                val point = DataPoint(id3.numAttributes)
                var next: String
                for (i in 0 until id3.numAttributes - 1) {
                    next = tokenizer.nextToken().trim { it <= ' ' }
                    if (id3.attributeTypes[i] == AttributeType.Continuous) {
                        if (next == "?" || next == "NaN") {
                            val value: Double = id3.meanValues[i]
                            point.attributes[i] = id3.getSymbolValue(i, value.toString())
                        } else {
                            try {
                                point.attributes[i] = id3.getSymbolValue(i, next)
                            } catch (_: Exception) {
                                System.err.println("Error reading continuous value in train data.")
                                exitProcess(1)
                            }
                        }
                    } else if (id3.attributeTypes[i] == AttributeType.Discrete) {
                        if (next == "?" || next == "NaN") {
                            point.attributes[i] = id3.mostCommonValues[i]
                        } else point.attributes[i] = id3.getSymbolValue(i, next)
                    }
                }
                //Check the created example against the random forest
                val classAttrValues = IntArray(id3.domainsIndexToValue[id3.classAttribute].size)
                val roots = id3.rootsRandomForest
                var node: TreeNode
                var resultClass = 0
                for (treeNode in roots) {
                    node = id3.testExamplePoint(point, treeNode)
                    //Check if the node is empty, if so, return its parent most frequent class.
                    var isEmpty = true
                    for (j in node.frequencyClasses.indices) {
                        if (node.frequencyClasses[j] != 0) {
                            isEmpty = false
                            break
                        }
                    }
                    //If node is empty
                    if (isEmpty) classAttrValues[id3.getMostCommonClass(node.parent)]++ else classAttrValues[id3.getMostCommonClass(node)]++
                }
                //Voting now
                for (i in 1 until classAttrValues.size) {
                    if (classAttrValues[i] > resultClass) resultClass = i
                }
                //Print line to output tmp file
                val classValue = id3.domainsIndexToValue[id3.classAttribute][resultClass]
                val line = "$input,$classValue"
                printOut.write(line)
                printOut.println()

                //continue the loop
                try {
                    input = bin.readLine()
                } catch (_: Exception) {
                    System.err.println("Unable to read line.")
                    exitProcess(1)
                }
            }
            printOut.close()
            print("\n")
            print("Results saved to tmp file.")
            print("\n")
        } else println("The forest file doesn't exist.")
    }

    fun queryRandomForest(file: String) {
        var fileLocal = file
        if (!fileLocal.endsWith(".forest")) fileLocal += ".forest"
        val inputFile = File(fileLocal)
        val `in` = Scanner(System.`in`)
        val id3: Cid3?
        if (inputFile.exists()) {
            id3 = deserializeFile(fileLocal)
            print("\n")
            print("Random Forest file of ${id3.numberOfTrees} trees deserialized.")
            print("\n")

            //Show version and time
            print(id3.createdWith)
            print("\n")

            //First display data statistics
            val sortedList: List<Triple<Int, Double, Double>> = if (id3.criteria == Criteria.Certainty)
                id3.attributeImportance.sortedWith(compareByDescending { it.second })
            else id3.attributeImportance.sortedWith(compareBy { it.second })

            //this is needed to format console output
            var longestString: String?
            longestString = ""
            for ((i, element) in sortedList.withIndex()){
                if (i > 99) break
                val attName: String = id3.attributeNames[element.first]
                if (longestString != null)
                    if (attName.length > longestString.length) longestString = attName
            }
            if (longestString!!.length < 14) longestString = "--------------"
            print("\n")
            //Print console output
            if (id3.criteria == Criteria.Certainty) {
                var fmt = "%1$10s %2$5s %3$1s %4$" + (longestString.length).toString() + "s%n"
                System.out.printf(fmt, "Importance", "Cause","", "Attribute Name")
                System.out.printf(fmt, "----------", "-----","", "--------------")
                for (i in sortedList.indices) {
                    if (i > 99) break
                    val rounded = String.format("%.2f", sortedList[i].second)
                    val isCause = if (sortedList[i].second - sortedList[i].third > 0) "yes"
                    else "no"
                    val fillerSize = longestString.length - id3.attributeNames[sortedList[i].first].length + 1
                    val filler = String(CharArray(fillerSize)).replace('\u0000', '·')
                    fmt = "%1$10s %2$5s %3$" + fillerSize.toString() + "s %4$" + id3.attributeNames[sortedList[i].first].length.toString() + "s%n"
                    System.out.printf(fmt, rounded, isCause, filler, id3.attributeNames[sortedList[i].first])
                }
            }
            else {
                val fmt = "%1$10s %2$" + (longestString.length + 10).toString() + "s%n"
                System.out.printf(fmt, "Importance", "Attribute Name")
                System.out.printf(fmt, "----------", "--------------")
                for (i in sortedList.indices) {
                    if (i > 99) break
                    val rounded = String.format("%.2f", sortedList[i].second)
                    System.out.printf(fmt, rounded, id3.attributeNames[sortedList[i].first])
                }
            }

            val example = DataPoint(id3.numAttributes)
            //Enter all attributes into an example except the class, that's why -1
            for (i in 0 until id3.numAttributes - 1) {
                if (id3.attributeTypes[i] == AttributeType.Ignore) continue
                if (id3.attributeTypes[i] == AttributeType.Discrete) {
                    print("\n")
                    print("Please enter attribute: " + id3.attributeNames[i])
                    print("\n")
                    print("(possible values are: ")
                    var values = StringBuilder()
                    val valuesArray = ArrayList<String>()
                    for (j in 0 until id3.domainsIndexToValue[i].size) {
                        if (id3.domainsIndexToValue[i][j] == "?") continue
                        valuesArray.add(id3.domainsIndexToValue[i][j] as String)
                    }
                    val comparator = NaturalOrderComparator()
                    val valuesList :List<String> = valuesArray.sortedWith(comparator).toList()
                    for (item in valuesList) {
                        values.append(item)
                        values.append(", ")
                    }
                    values = StringBuilder("?, " + values.substring(0, values.length - 2))
                    print("$values)")
                    print("\n")
                    while (true) {
                        var value: Int
                        val s = `in`.nextLine()
                        if (s == "?") {
                            value = id3.mostCommonValues[i]
                            example.attributes[i] = value
                            break
                        } else if (id3.domainsIndexToValue[i].containsValue(s)) {
                            example.attributes[i] = id3.getSymbolValue(i, s)
                            break
                        } else println("Please enter a valid value:")
                    }
                } else {
                    print("\n")
                    print("Please enter attribute: " + id3.attributeNames[i])
                    print("\n")
                    while (true) {
                        val s = `in`.nextLine()
                        try {
                            val value: Double = if (s == "?") id3.meanValues[i] else s.toDouble()
                            example.attributes[i] = id3.getSymbolValue(i, value.toString())
                            break
                        } catch (_: Exception) {
                            println("Please enter a valid value:")
                        }
                    }
                }
            }

            //Check the created example against the random forest
            val classAttrValues = IntArray(id3.domainsIndexToValue[id3.classAttribute].size)
            val roots = id3.rootsRandomForest
            var node: TreeNode
            var resultClass = 0
            for (treeNode in roots) {
                node = id3.testExamplePoint(example, treeNode)
                //Check if the node is empty, if so, return its parent most frequent class.
                var isEmpty = true
                for (j in node.frequencyClasses.indices) {
                    if (node.frequencyClasses[j] != 0) {
                        isEmpty = false
                        break
                    }
                }
                //If node is empty
                if (isEmpty) classAttrValues[id3.getMostCommonClass(node.parent)]++ else classAttrValues[id3.getMostCommonClass(node)]++
            }
            //Voting now
            for (i in 1 until classAttrValues.size) {
                if (classAttrValues[i] > resultClass) resultClass = i
            }
            //Print the answer
            val mostCommonStr = id3.domainsIndexToValue[id3.classAttribute][resultClass]
            print("\n")
            println("Class attribute value is: $mostCommonStr")
        } else println("The file doesn't exist.")
    }

    private fun formatDuration(duration: Duration): String {
        val seconds = duration.seconds
        val absSeconds = abs(seconds)
        val positive = String.format(
                "%d:%02d:%02d",
                absSeconds / 3600,
                absSeconds % 3600 / 60,
                absSeconds % 60)
        return if (seconds < 0) "-$positive" else positive
    }

    fun playSound() {
        try {
            val audioInputStream = AudioSystem.getAudioInputStream(
                this.javaClass.getResource("breaks.wav")
            )
            val clip = AudioSystem.getClip()
            clip.open(audioInputStream)
            clip.start()
            //The next lines is needed for the program to play the whole sound, otherwise it plays just a bit and exits.
            Thread.sleep(2000)
        } catch (ex: java.lang.Exception) {
            print(ex.toString())
        }
    }

    companion object {
        private const val serialVersionUID: Long = 42L
        //Here is the definition of the main function
        @JvmStatic
        fun main(args: Array<String>) {
            val me = Cid3()

            //Initialize values
            me.isCrossValidation = false
            me.testDataExists = false
            me.isRandomForest = false

            //Create command line options
            val options = Options()
            val analysis = Option("a", "analysis", true, "show causal analysis report")
            analysis.isRequired = false
            analysis.setOptionalArg(true)
            analysis.argName = "name"
            options.addOption(analysis)
            val help = Option("h", "help", false, "print this message")
            help.isRequired = false
            options.addOption(help)
            val version = Option("ver", "version", false, "version")
            version.isRequired = false
            options.addOption(version)
            val save = Option("s", "save", false, "save tree/random forest")
            save.isRequired = false
            options.addOption(save)
            val partition = Option("p", "partition", false, "partition train/test data")
            save.isRequired = false
            options.addOption(partition)
            val file = Option("f", "file", true, "input file")
            file.isRequired = true
            file.argName = "name"
            options.addOption(file)
            val crossValidation = Option("v", "validation", false, "create 10-fold cross-validation")
            crossValidation.isRequired = false
            options.addOption(crossValidation)
            val randomForest = Option("r", "forest", true, "create random forest, enter # of trees")
            randomForest.isRequired = false
            randomForest.argName = "amount"
            options.addOption(randomForest)
            val query = Option("q", "query", true, "query model, enter: t[Tree] or r[Random forest]")
            query.isRequired = false
            query.argName = "type"
            options.addOption(query)
            val output = Option("o", "output", true, "output file")
            output.isRequired = false
            output.argName = "name"
            options.addOption(output)
            val threads = Option("t", "threads", true, "maximum number of threads (default is 500)")
            threads.isRequired = false
            threads.argName = "amount"
            options.addOption(threads)

            //Declare parser and formatter
            val parser: CommandLineParser = DefaultParser()
            val formatter = HelpFormatter()
            val cmd: CommandLine
            try {
                for (arg in args) {
                    if (arg == "-ver" || arg == "--version") {
                        //Print version
                        print("CID3 version: ${me.version}")
                        print("\n")
                        exitProcess(0)
                    }
                }
                for (arg in args) {
                    if (arg == "-h" || arg == "--help") {
                        //Print help message
                        formatter.printHelp("java -jar cid3.jar", options)
                        exitProcess(0)
                    }
                }
                cmd = parser.parse(options, args)
            } catch (e: ParseException) {
                println(e.message)
                formatter.printHelp("java -jar cid3.jar", options)
                exitProcess(1)
            }

            //Set number of threads
            if (cmd.hasOption("threads")) {
                try {
                    val intThreads = cmd.getOptionValue("threads").toInt()
                    me.maxThreads = intThreads
                }
                catch (_:java.lang.Exception){
                    print("Error: Incorrect number of threads format")
                    print("\n")
                    exitProcess(1)
                }
            }

            //Set criteria
            me.criteria = Criteria.Certainty

            //Set file path
            var inputFilePath = cmd.getOptionValue("file")
            val originalInputFilePath = inputFilePath

            //Set validation
            me.isCrossValidation = cmd.hasOption("validation")

            //Set split data
            me.splitTrainData = cmd.hasOption("partition")

            //Set Random Forest
            if (cmd.hasOption("forest")) {
                val numberOfTrees = cmd.getOptionValue("forest")
                try {
                    me.numberOfTrees = numberOfTrees.toInt()
                } catch (_: Exception) {
                    print("Error: Incorrect number of trees")
                    print("\n")
                    exitProcess(1)
                }
                me.isRandomForest = true
            } else me.isRandomForest = false

            //Show application title
            print("\n")
            val dayInMonth = LocalDateTime.now().dayOfMonth
            var day = LocalDateTime.now().dayOfWeek.name.lowercase(Locale.getDefault())
            day = day.substring(0, 1).uppercase(Locale.getDefault()) + day.substring(1)
            var month = LocalDateTime.now().month.name.lowercase(Locale.getDefault())
            month = month.substring(0, 1).uppercase(Locale.getDefault()) + month.substring(1)
            val time = DateTimeFormatter.ofPattern("hh:mm:ss a")
            val timeString = LocalDateTime.now().format(time)
            val year = LocalDateTime.now().year.toString()
            print("CID3 [Version ${me.version}]              $day $month $dayInMonth, $year $timeString")
            print("\n")
            print("--------------------")
            print("\n")

            //Check for incompatible options
            if (cmd.hasOption("analysis")){
                if (cmd.hasOption("validation") || cmd.hasOption("save")
                    || cmd.hasOption("forest") || cmd.hasOption("criteria") || cmd.hasOption("partition")
                    || cmd.hasOption("output")){
                    System.err.println("Options -v, -s, -c, -r, -p, -o are not compatible with -a.")
                    exitProcess(1)
                }
            }
            if (cmd.hasOption("query")){
                if (cmd.hasOption("validation") || cmd.hasOption("save")
                        || cmd.hasOption("forest") || cmd.hasOption("criteria") || cmd.hasOption("partition")){
                    System.err.println("Options -v, -s, -c, -r, -p are not compatible with -q.")
                    exitProcess(1)
                }
            }
            if (cmd.hasOption("threads")){
                if (cmd.hasOption("validation") || cmd.hasOption("forest")){
                    System.err.println("Options -v and -r are not compatible with -t.")
                    exitProcess(1)
                }
            }

            if (cmd.hasOption("save")){
                if (cmd.hasOption("validation") || cmd.hasOption("partition")){
                    System.err.println("Options -v and -p are not compatible with -s.")
                    exitProcess(1)
                }
            }
            if (cmd.hasOption("output")){
                if (cmd.hasOption("validation") || cmd.hasOption("forest") || cmd.hasOption("criteria")
                        || cmd.hasOption("partition")){
                    System.err.println("Options -v, -r, -p and -c are not compatible with -o.")
                    exitProcess(1)
                }
            }

            //if is a query, deserialize file and query model
            val outputFilePath: String
            if (cmd.hasOption("query")) {
                val model = cmd.getOptionValue("query")
                if (model == "t") {
                    if (cmd.hasOption("output")) {
                        outputFilePath = cmd.getOptionValue("output")
                        me.queryTreeOutput(originalInputFilePath, outputFilePath)
                    } else me.queryTree(originalInputFilePath)
                } else if (model == "r") {
                    if (cmd.hasOption("output")) {
                        outputFilePath = cmd.getOptionValue("output")
                        me.queryRandomForestOutput(originalInputFilePath, outputFilePath)
                    } else me.queryRandomForest(originalInputFilePath)
                }
            }
            else {
                //Check if test data exists
                if (!inputFilePath.endsWith(".data")) inputFilePath += ".data"
                var nameTestData = inputFilePath.substring(0, inputFilePath.length - 4)
                nameTestData += "test"
                val inputFile = File(nameTestData)
                me.testDataExists = inputFile.exists()

                //Set global variable
                me.fileName = inputFilePath

                //Read names file
                me.readNames(inputFilePath)

                //Print animation for data reading
                val consoleHelperReading = ConsoleHelper()
                val threadReading = Thread {
                    while(me.runAnimationReading) {
                        consoleHelperReading.animate()
                        Thread.sleep(500)
                    }
                }
                threadReading.priority = Thread.MAX_PRIORITY
                threadReading.start()
                //Read data
                me.readData(inputFilePath)

                //Print animation for data reading
                val consoleHelperReadingTest = ConsoleHelper()
                val threadReadingTest = Thread {
                    while(me.runAnimationReadingTest) {
                        consoleHelperReadingTest.animate()
                        Thread.sleep(500)
                    }
                }
                //Read test data
                if (me.testDataExists) {
                    threadReadingTest.priority = Thread.MAX_PRIORITY
                    threadReadingTest.start()
                    me.readTestData(nameTestData)
                }

                //Initialize falsePositives and falseNegatives
                me.falsePositivesTrain = IntArray(me.domainsIndexToValue[me.numAttributes - 1].size)
                me.falseNegativesTrain = IntArray(me.domainsIndexToValue[me.numAttributes - 1].size)
                me.falsePositivesTest = IntArray(me.domainsIndexToValue[me.numAttributes - 1].size)
                me.falseNegativesTest = IntArray(me.domainsIndexToValue[me.numAttributes - 1].size)

                //Initialize classes number of cases
                me.classNoOfCasesTrain = IntArray(me.domainsIndexToValue[me.numAttributes - 1].size)
                me.classNoOfCasesTest = IntArray(me.domainsIndexToValue[me.numAttributes - 1].size)
                me.countClasses()

                if (cmd.hasOption("analysis")) {
                    //Print animation for creating analysis report
                    val consoleHelperCalculating = ConsoleHelper()
                    val threadCalculating = Thread {
                        while(me.runAnimationAnalysis) {
                            consoleHelperCalculating.animate()
                            Thread.sleep(500)
                        }
                    }
                    threadCalculating.priority = Thread.MAX_PRIORITY
                    threadCalculating.start()

                    me.trainData.addAll(me.testData)
                    me.root.data = me.trainData
                    me.testData.clear()
                    me.setMeanValues()
                    me.setMostCommonValues()
                    me.imputeMissing()
                    var attName = ""
                    if(cmd.getOptionValue("analysis") != null) {
                        attName = cmd.getOptionValue("analysis")
                        if (!me.attributeNames.contains(attName)){
                            print("\n")
                            print("Error: Unknown attribute name.")
                            print("\n")
                            exitProcess(1)
                        }
                    }
                    me.createCausalAnalysisReport(attName)
                }//Create a Tree or a Random Forest for saving to disk
                else if (cmd.hasOption("save")) {
                    //Print animation for creating calculations
                    val consoleHelperCalculating = ConsoleHelper()
                    val threadCalculating = Thread {
                        while(me.runAnimationCalculating) {
                            consoleHelperCalculating.animate()
                            Thread.sleep(500)
                        }
                    }
                    threadCalculating.priority = Thread.MAX_PRIORITY
                    threadCalculating.start()
                    me.save = true
                    me.trainData.addAll(me.testData)
                    me.root.data = me.trainData
                    me.testData.clear()
                    me.setMeanValues()
                    me.setMostCommonValues()
                    me.imputeMissing()
                    if (me.isRandomForest) {
                        me.createRandomForest(me.root.data, me.rootsRandomForest, false)
                        print("\n")
                        print("Saving random forest...")
                        me.createFileRF()
                        print("\n")
                        print("Random forest saved to disk.")
                    } else {
                        me.createDecisionTree()
                        print("\n")
                        print("Saving tree...")
                        me.createFile()
                        print("\n")
                        print("Tree saved to disk.")
                    }
                    print("\n")
                } //Create CV, RF, Tree without saving
                else {
                    //Print animation for creating calculations
                    val consoleHelperCalculating = ConsoleHelper()
                    val threadCalculating = Thread {
                        while(me.runAnimationCalculating) {
                            consoleHelperCalculating.animate()
                            Thread.sleep(500)
                        }
                    }
                    threadCalculating.priority = Thread.MAX_PRIORITY
                    threadCalculating.start()
                    me.setMeanValues()
                    me.setMostCommonValues()
                    me.imputeMissing()
                    if (me.isCrossValidation && me.isRandomForest) me.createCrossValidationRF()
                    else if (me.isCrossValidation) me.createCrossValidation()
                    else if (me.isRandomForest) me.createRandomForest(me.root.data, me.rootsRandomForest, false)
                    else me.createDecisionTree()
                }
                me.playSound()
            }
        }
    }
}
