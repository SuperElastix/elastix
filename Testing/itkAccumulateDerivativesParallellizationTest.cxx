/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#include "itkSmartPointer.h"
#include "itkArray.h"
#include <vector>
#include <algorithm>
#include <iomanip>
#include "itkNumericTraits.h"

// Report timings
#include <ctime>
#include "itkTimeProbe.h"
#include "itkTimeProbesCollectorBase.h"

// Multi-threading using ITK threads
#include "itkMultiThreaderBase.h"

#include <cassert>

// select double or float internal type of array
#if 0
typedef float InternalScalarType;
#else
typedef double InternalScalarType;
#endif
typedef unsigned int ThreadIdType;

class MetricTEMP : public itk::Object
{
public:
  /** Standard class typedefs. */
  using Self = MetricTEMP;
  using Pointer = itk::SmartPointer<Self>;
  itkNewMacro(Self);

  using DerivativeValueType = InternalScalarType;
  using DerivativeType = itk::Array<DerivativeValueType>;

  unsigned long                       m_NumberOfParameters;
  mutable std::vector<DerivativeType> m_ThreaderDerivatives;

  using ThreadInfoType = itk::MultiThreaderBase::WorkUnitInfo;

  itk::MultiThreaderBase::Pointer m_Threader;
  DerivativeValueType             m_NormalSum;
  ThreadIdType                    m_NumberOfThreads;
  bool                            m_UseMultiThreaded;

  struct MultiThreaderParameterType
  {
    // To give the threads access to all members.
    Self * st_Metric;
    // Used for accumulating derivatives
    DerivativeValueType * st_DerivativePointer;
    DerivativeValueType   st_NormalizationFactor;
  };
  mutable MultiThreaderParameterType m_ThreaderMetricParameters;

  // Constructor
  MetricTEMP()
  {
    this->m_ThreaderMetricParameters.st_Metric = nullptr;
    this->m_ThreaderMetricParameters.st_DerivativePointer = nullptr;
    this->m_ThreaderMetricParameters.st_NormalizationFactor = 0.0;

    this->m_NumberOfParameters = 0;
    this->m_Threader = itk::MultiThreaderBase::New();
    this->m_NumberOfThreads = this->m_Threader->GetNumberOfWorkUnits();
    this->m_UseMultiThreaded = false;
    this->m_NormalSum = 3.1415926;
  }


  void
  AccumulateDerivatives(DerivativeType & derivative)
  {
    DerivativeValueType normal_sum = this->m_NormalSum;
    this->m_NumberOfThreads = this->m_Threader->GetNumberOfWorkUnits();

    /** Accumulate derivatives. */
    if (!this->m_UseMultiThreaded) // single threadedly
    {
      derivative = this->m_ThreaderDerivatives[0] * normal_sum;
      for (ThreadIdType i = 1; i < this->m_NumberOfThreads; ++i)
      {
        derivative += this->m_ThreaderDerivatives[i] * normal_sum;
      }
    }
    // compute multi-threadedly with itk threads
    else
    {
      this->m_ThreaderMetricParameters.st_Metric = this;
      this->m_ThreaderMetricParameters.st_DerivativePointer = derivative.begin();
      this->m_ThreaderMetricParameters.st_NormalizationFactor = 1.0 / normal_sum;

      this->m_Threader->SetSingleMethodAndExecute(this->AccumulateDerivativesThreaderCallback,
                                                  &this->m_ThreaderMetricParameters);
    }
  } // end AccumulateDerivatives()


  /**
   *********** AccumulateDerivativesThreaderCallback *************
   */

  static ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
  AccumulateDerivativesThreaderCallback(void * arg)
  {
    assert(arg);
    const auto & infoStruct = *static_cast<ThreadInfoType *>(arg);
    ThreadIdType threadID = infoStruct.WorkUnitID;
    ThreadIdType nrOfThreads = infoStruct.NumberOfWorkUnits;

    assert(infoStruct.UserData);
    const auto & userData = *static_cast<MultiThreaderParameterType *>(infoStruct.UserData);

    assert(userData.st_Metric);
    Self & metric = *(userData.st_Metric);

    const unsigned int numPar = metric.m_NumberOfParameters;
    const auto         subSize =
      static_cast<unsigned int>(std::ceil(static_cast<double>(numPar) / static_cast<double>(nrOfThreads)));
    const unsigned int jmin = threadID * subSize;
    const unsigned int jmax = std::min((threadID + 1) * subSize, numPar);

    for (unsigned int j = jmin; j < jmax; ++j)
    {
      DerivativeValueType sum{};
      for (ThreadIdType i = 0; i < nrOfThreads; ++i)
      {
        sum += metric.m_ThreaderDerivatives[i][j];
      }
      userData.st_DerivativePointer[j] = sum / userData.st_NormalizationFactor;
    }

    return ITK_THREAD_RETURN_DEFAULT_VALUE;

  } // end AccumulateDerivativesThreaderCallback()
};

// end class Metric

//-------------------------------------------------------------------------------------

int
main()
{
  // Declare and setup
  std::cout << std::fixed << std::showpoint << std::setprecision(8);
  std::cout << "RESULTS FOR InternalScalarType = " << typeid(InternalScalarType).name() << "\n\n" << std::endl;

  /** Typedefs. */
  using MetricClass = MetricTEMP;
  using DerivativeType = MetricClass::DerivativeType;

  auto metric = MetricClass::New();

  // test parameters
  std::vector<unsigned int> arraySizes;
  arraySizes.push_back(1e2);
  arraySizes.push_back(1e3);
  arraySizes.push_back(1e4);
  arraySizes.push_back(1e5);
  arraySizes.push_back(1e6);
  arraySizes.push_back(1e7);
  std::vector<unsigned int> repetitions;
  repetitions.push_back(2e6);
  repetitions.push_back(2e5);
  repetitions.push_back(2e4);
  repetitions.push_back(2e3);
  repetitions.push_back(1e2);
  repetitions.push_back(1e1);

  const ThreadIdType nrThreads = metric->m_Threader->GetNumberOfWorkUnits();

  /** For all sizes. */
  for (unsigned int s = 0; s < arraySizes.size(); ++s)
  {
    std::cout << "Array size = " << arraySizes[s] << std::endl;

    /** Setup. */
    itk::TimeProbesCollectorBase timeCollector;
    unsigned int                 rep = 0;
    repetitions[s] = 1; // outcomment this line for full testing

    DerivativeType derivative(arraySizes[s], 0.0);

    metric->m_ThreaderDerivatives.resize(nrThreads);
    metric->m_NumberOfParameters = arraySizes[s];
    for (ThreadIdType t = 0; t < nrThreads; ++t)
    {
      // Allocate
      metric->m_ThreaderDerivatives[t].SetSize(metric->m_NumberOfParameters);
      metric->m_ThreaderDerivatives[t].Fill(0);

      for (unsigned int i = 0; i < arraySizes[s]; ++i)
      {
        metric->m_ThreaderDerivatives[t][i] = 2.1;
      }
    }

    /** Time the single-threaded implementation. */
    metric->m_UseMultiThreaded = false;
    for (unsigned int i = 0; i < repetitions[s]; ++i)
    {
      timeCollector.Start("st");
      metric->AccumulateDerivatives(derivative);
      timeCollector.Stop("st");
    }

    /** Time the ITK multi-threaded implementation. */
    metric->m_UseMultiThreaded = true;
    if (arraySizes[s] < 5000)
    {
      rep = repetitions[s] / 100.0;
    }
    else
    {
      rep = repetitions[s];
    }
    if (rep < 10)
    {
      rep = 10;
    }
    for (unsigned int i = 0; i < rep; ++i)
    {
      timeCollector.Start("ITK (mt)");
      metric->AccumulateDerivatives(derivative);
      timeCollector.Stop("ITK (mt)");
    }

    /** Report timings for this array size. */
    timeCollector.Report();
    std::cout << std::endl;

  } // end loop over array sizes

  return EXIT_SUCCESS;

} // end main
