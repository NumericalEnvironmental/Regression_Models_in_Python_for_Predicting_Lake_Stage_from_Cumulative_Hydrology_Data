# Regression_Models_in_Python_for_Predicting_Lake_Stage_from_Cumulative_Hydrology_Data

![Preview](https://numericalenvironmental.files.wordpress.com/2021/06/svr-validation.png)

This is a short Python script intended for analysis of historic rainfall data and lake stage hydrographs, with the objective of predicting the latter from the former. Specifically, the script trains linear, random forest, support vector, and multilayer perceptron regression models to represent relationships between cumulative rainfall deficits, averaged over time periods ranging from months to years, and hydrograph data from a lake in an urban environment. The regression models are then applied to a validation data set where lake stage is predicted by subsequent cumulative rainfall deficits. See my blog [https://numericalenvironmental.wordpress.com/2021/06/13/predicting-lake-stage-in-an-urban-environment-using-historic-rainfall-data-and-candidate-regression-models/] for more details of both the approach and the example application.
NumPy, Scikit-Learn, Matplotlib, and miscellaneous SciPy classes are required to run the script, as are the following input files:
* precip_record.csv – rainfall record, by month
* hydrograph.csv – lake stage, by date (will be interpolated to correspond to monthly rainfall record)
* params.txt – miscellaneous parameters, including bounding dates for correlation analysis, date for splitting training and validation data sets, time intervals for backward-looking cumulative rainfall deficit, posited mean annual rainfall

I'd appreciate hearing back from you if you find the code useful. Questions or comments are welcome at walt.mcnab@gmail.com.

THIS CODE/SOFTWARE IS PROVIDED IN SOURCE OR BINARY FORM "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
