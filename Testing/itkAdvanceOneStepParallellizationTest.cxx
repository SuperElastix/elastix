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

// Report timings
#include <ctime>
#include "itkTimeProbe.h"
#include "itkTimeProbesCollectorBase.h"

// Multi-threading using ITK threads
#include "itkMultiThreaderBase.h"

// Single-threaded vector arithmetic using Eigen
#ifdef ELASTIX_USE_EIGEN
#  include <Eigen/Dense>
#  include <Eigen/Core>
#endif

#include <cassert>

// select double or float internal type of array
#if 0
typedef float InternalScalarType;
#else
typedef double InternalScalarType;
#endif

#ifdef ELASTIX_USE_EIGEN
#  if 0
typedef Eigen::VectorXf ParametersTypeEigen;
#  else
typedef Eigen::VectorXd ParametersTypeEigen;
#  endif
#endif

class OptimizerTEMP : public itk::Object
{
public:
  /** Standard class typedefs. */
  using Self = OptimizerTEMP;
  using Pointer = itk::SmartPointer<Self>;
  itkNewMacro(Self);

  using ParametersType = itk::Array<InternalScalarType>;

  unsigned long      m_NumberOfParameters;
  ParametersType     m_CurrentPosition;
  ParametersType     m_Gradient;
  InternalScalarType m_LearningRate;

  using ThreadInfoType = itk::MultiThreaderBase::WorkUnitInfo;
  itk::MultiThreaderBase::Pointer m_Threader;
  bool                            m_UseEigen;
  bool                            m_UseMultiThreaded;

  struct MultiThreaderParameterType
  {
    ParametersType * t_NewPosition;
    Self *           t_Optimizer;
  };

  OptimizerTEMP()
  {
    this->m_NumberOfParameters = 0;
    this->m_LearningRate = 0.0;
    this->m_Threader = itk::MultiThreaderBase::New();
    this->m_Threader->SetNumberOfWorkUnits(8);
    this->m_UseEigen = false;
    this->m_UseMultiThreaded = false;
  }


  void
  AdvanceOneStep()
  {
    const unsigned int spaceDimension = m_NumberOfParameters;
    ParametersType &   newPosition = this->m_CurrentPosition;

    if (!this->m_UseMultiThreaded)
    {
      /** Get a pointer to the current position. */
      const InternalScalarType * currentPosition = this->m_CurrentPosition.data_block();
      const double               learningRate = this->m_LearningRate;
      const InternalScalarType * gradient = this->m_Gradient.data_block();
      InternalScalarType *       newPos = newPosition.data_block();

      /** Update the new position. */
      for (unsigned int j = 0; j < spaceDimension; ++j)
      {
        // newPosition[j] = currentPosition[j] - this->m_LearningRate * this->m_Gradient[j];
        newPos[j] = currentPosition[j] - learningRate * gradient[j];
      }
    }
#ifdef ELASTIX_USE_EIGEN
    else if (this->m_UseEigen)
    {
      /** Get a reference to the current position. */
      const ParametersType &   currentPosition = this->m_CurrentPosition;
      const InternalScalarType learningRate = this->m_LearningRate;

      /** Wrap itk::Arrays into Eigen jackets. */
      Eigen::Map<ParametersTypeEigen>       newPositionE(newPosition.data_block(), spaceDimension);
      Eigen::Map<const ParametersTypeEigen> currentPositionE(currentPosition.data_block(), spaceDimension);
      Eigen::Map<ParametersTypeEigen>       gradientE(this->m_Gradient.data_block(), spaceDimension);

      /** Update the new position. */
      // Eigen::setNbThreads( this->m_Threader->GetNumberOfWorkUnits() );
      newPositionE = currentPositionE - learningRate * gradientE;
    }
#endif
    else
    {
      /** Fill the threader parameter struct with information. */
      MultiThreaderParameterType userData;
      userData.t_NewPosition = &newPosition;
      userData.t_Optimizer = this;

      /** Call multi-threaded AdvanceOneStep(). */
      this->m_Threader->SetSingleMethodAndExecute(AdvanceOneStepThreaderCallback, &userData);
    }
  } // end


  /** The callback function. */
  static ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
  AdvanceOneStepThreaderCallback(void * arg)
  {
    /** Get the current thread id and user data. */
    assert(arg);
    const auto &      infoStruct = *static_cast<ThreadInfoType *>(arg);
    itk::ThreadIdType threadID = infoStruct.WorkUnitID;

    assert(infoStruct.UserData);
    const auto & userData = *static_cast<MultiThreaderParameterType *>(infoStruct.UserData);

    /** Call the real implementation. */
    userData.t_Optimizer->ThreadedAdvanceOneStep2(threadID, *(userData.t_NewPosition));

    return ITK_THREAD_RETURN_DEFAULT_VALUE;

  } // end AdvanceOneStepThreaderCallback()


  /** The threaded implementation of AdvanceOneStep(). */
  inline void
  ThreadedAdvanceOneStep(itk::ThreadIdType threadId, ParametersType & newPosition)
  {
    /** Compute the range for this thread. */
    const unsigned int spaceDimension = m_NumberOfParameters;
    const auto         subSize = static_cast<unsigned int>(
      std::ceil(static_cast<double>(spaceDimension) / static_cast<double>(this->m_Threader->GetNumberOfWorkUnits())));
    const unsigned int jmin = threadId * subSize;
    const unsigned int jmax = std::min((threadId + 1) * subSize, spaceDimension);

    /** Get a reference to the current position. */
    const ParametersType & currentPosition = this->m_CurrentPosition;
    const double           learningRate = this->m_LearningRate;
    const ParametersType & gradient = this->m_Gradient;

    /** Advance one step: mu_{k+1} = mu_k - a_k * gradient_k */
    for (unsigned int j = jmin; j < jmax; ++j)
    {
      newPosition[j] = currentPosition[j] - learningRate * gradient[j];
    }

  } // end ThreadedAdvanceOneStep()


  /** The threaded implementation of AdvanceOneStep(). */
  inline void
  ThreadedAdvanceOneStep2(itk::ThreadIdType threadId, ParametersType & newPosition)
  {
    /** Compute the range for this thread. */
    const unsigned int spaceDimension = m_NumberOfParameters;
    const auto         subSize = static_cast<unsigned int>(
      std::ceil(static_cast<double>(spaceDimension) / static_cast<double>(this->m_Threader->GetNumberOfWorkUnits())));
    const unsigned int jmin = threadId * subSize;
    const unsigned int jmax = std::min((threadId + 1) * subSize, spaceDimension);

    /** Get a pointer to the current position. */
    const InternalScalarType * currentPosition = this->m_CurrentPosition.data_block();
    const double               learningRate = this->m_LearningRate;
    const InternalScalarType * gradient = this->m_Gradient.data_block();
    InternalScalarType *       newPos = newPosition.data_block();

    /** Advance one step: mu_{k+1} = mu_k - a_k * gradient_k */
    for (unsigned int j = jmin; j < jmax; ++j)
    {
      newPos[j] = currentPosition[j] - learningRate * gradient[j];
    }

  } // end ThreadedAdvanceOneStep()
};

// end class Optimizer

//-------------------------------------------------------------------------------------

int
main()
{
  // Declare and setup
  std::cout << std::fixed << std::showpoint << std::setprecision(8);
  std::cout << "RESULTS FOR InternalScalarType = " << typeid(InternalScalarType).name() << "\n\n" << std::endl;

  /** Typedefs. */
  using OptimizerClass = OptimizerTEMP;
  using ParametersType = OptimizerClass::ParametersType;

  auto optimizer = OptimizerClass::New();

  // test parameters
  std::vector<unsigned int> arraySizes;
  arraySizes.push_back(1e2);
  arraySizes.push_back(1e3);
  arraySizes.push_back(1e4);
  arraySizes.push_back(1e5);
  arraySizes.push_back(1e6);
  arraySizes.push_back(1e7);
  std::vector<unsigned int> repetitions;
  repetitions.push_back(2e7);
  repetitions.push_back(2e6);
  repetitions.push_back(2e5);
  repetitions.push_back(2e4);
  repetitions.push_back(1e3);
  repetitions.push_back(1e2);

  /** For all sizes. */
  for (unsigned int s = 0; s < arraySizes.size(); ++s)
  {
    std::cout << "Array size = " << arraySizes[s] << std::endl;

    /** Setup. */
    itk::TimeProbesCollectorBase timeCollector;
    repetitions[s] = 1; // outcomment this line for full testing

    ParametersType newPos(arraySizes[s]);
    ParametersType curPos(arraySizes[s]);
    ParametersType gradient(arraySizes[s]);
    for (unsigned int i = 0; i < arraySizes[s]; ++i)
    {
      curPos[i] = 2.1;
      gradient[i] = 2.1;
    }
    optimizer->m_NumberOfParameters = arraySizes[s];
    optimizer->m_LearningRate = 3.67;
    optimizer->m_CurrentPosition = curPos;
    optimizer->m_Gradient = gradient;

    /** Time the ITK single-threaded implementation. */
    optimizer->m_UseEigen = false;
    optimizer->m_UseMultiThreaded = false;
    for (unsigned int i = 0; i < repetitions[s]; ++i)
    {
      timeCollector.Start("st");
      optimizer->AdvanceOneStep();
      timeCollector.Stop("st");
    }

    /** Time the ITK multi-threaded implementation. */
    optimizer->m_UseEigen = false;
    optimizer->m_UseMultiThreaded = true;
    unsigned int rep = repetitions[s] / 1000.0;
    if (rep < 10)
    {
      rep = 10;
    }
    for (unsigned int i = 0; i < rep; ++i)
    {
      timeCollector.Start("ITK (mt)");
      optimizer->AdvanceOneStep();
      timeCollector.Stop("ITK (mt)");
    }

    /** Time the Eigen single-threaded implementation. */
#ifdef ELASTIX_USE_EIGEN
    optimizer->m_UseEigen = true;
    optimizer->m_UseMultiThreaded = true;
    for (unsigned int i = 0; i < repetitions[s]; ++i)
    {
      timeCollector.Start("Eigen (st)");
      optimizer->AdvanceOneStep();
      timeCollector.Stop("Eigen (st)");
    }
#endif

    // Report timings for this array size
    timeCollector.Report(std::cout, false, true);
    std::cout << std::endl;

  } // end loop over array sizes

  return EXIT_SUCCESS;

} // end main
