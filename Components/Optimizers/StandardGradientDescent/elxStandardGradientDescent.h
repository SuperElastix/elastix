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
#ifndef elxStandardGradientDescent_h
#define elxStandardGradientDescent_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkStandardGradientDescentOptimizer.h"

namespace elastix
{

/**
 * \class StandardGradientDescent
 * \brief A gradient descent optimizer with a decaying gain.
 *
 * This class is a wrap around the StandardGradientDescentOptimizer class.
 * It takes care of setting parameters and printing progress information.
 * For more information about the optimisation method, please read the documentation
 * of the StandardGradientDescentOptimizer class.
 *
 * The parameters used in this class are:
 * \parameter Optimizer: Select this optimizer as follows:\n
 *   <tt>(Optimizer "StandardGradientDescent")</tt>
 * \parameter MaximumNumberOfIterations: The maximum number of iterations in each resolution. \n
 *   example: <tt>(MaximumNumberOfIterations 100 100 50)</tt> \n
 *    Default/recommended value: 500.
 * \parameter MaximumNumberOfSamplingAttempts: The maximum number of sampling attempts. Sometimes
 *   not enough corresponding samples can be drawn, upon which an exception is thrown. With this
 *   parameter it is possible to try to draw another set of samples. \n
 *   example: <tt>(MaximumNumberOfSamplingAttempts 10 15 10)</tt> \n
 *    Default value: 0, i.e. just fail immediately, for backward compatibility.
 * \parameter SP_a: The gain \f$a(k)\f$ at each iteration \f$k\f$ is defined by \n
 *   \f$a(k) =  SP\_a / (SP\_A + k + 1)^{SP\_alpha}\f$. \n
 *   SP_a can be defined for each resolution. \n
 *   example: <tt>(SP_a 3200.0 3200.0 1600.0)</tt> \n
 *   The default value is 400.0. Tuning this variable for you specific problem is recommended.
 * \parameter SP_A: The gain \f$a(k)\f$ at each iteration \f$k\f$ is defined by \n
 *   \f$a(k) =  SP\_a / (SP\_A + k + 1)^{SP\_alpha}\f$. \n
 *   SP_A can be defined for each resolution. \n
 *   example: <tt>(SP_A 50.0 50.0 100.0)</tt> \n
 *   The default/recommended value is 50.0.
 * \parameter SP_alpha: The gain \f$a(k)\f$ at each iteration \f$k\f$ is defined by \n
 *   \f$a(k) =  SP\_a / (SP\_A + k + 1)^{SP\_alpha}\f$. \n
 *   SP_alpha can be defined for each resolution. \n
 *   example: <tt>(SP_alpha 0.602 0.602 0.602)</tt> \n
 *   The default/recommended value is 0.602.
 *
 * \sa StandardGradientDescentOptimizer
 * \ingroup Optimizers
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT StandardGradientDescent
  : public itk::StandardGradientDescentOptimizer
  , public OptimizerBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(StandardGradientDescent);

  /** Standard ITK.*/
  using Self = StandardGradientDescent;
  using Superclass1 = StandardGradientDescentOptimizer;
  using Superclass2 = OptimizerBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(StandardGradientDescent, StandardGradientDescentOptimizer);

  /** Name of this class.
   * Use this name in the parameter file to select this specific optimizer.
   * example: <tt>(Optimizer "StandardGradientDescent")</tt>\n
   */
  elxClassNameMacro("StandardGradientDescent");

  /** Typedef's inherited from Superclass1, the StandardGradientDescentOptimizer.*/
  using Superclass1::CostFunctionType;
  using Superclass1::CostFunctionPointer;
  using Superclass1::StopConditionType;

  /** Typedef's inherited from Superclass2, the elastix OptimizerBase .*/
  using typename Superclass2::ElastixType;
  using typename Superclass2::RegistrationType;
  using ITKBaseType = typename Superclass2::ITKBaseType;

  /** Typedef for the ParametersType. */
  using typename Superclass1::ParametersType;

  /** Methods invoked by elastix, in which parameters can be set and
   * progress information can be printed. */
  void
  BeforeRegistration() override;

  void
  BeforeEachResolution() override;

  void
  AfterEachResolution() override;

  void
  AfterEachIteration() override;

  void
  AfterRegistration() override;

  /** Check if any scales are set, and set the UseScales flag on or off;
   * after that call the superclass' implementation */
  void
  StartOptimization() override;

  /** Stop optimisation and pass on exception. */
  void
  MetricErrorResponse(itk::ExceptionObject & err) override;

  /** Add SetCurrentPositionPublic, which calls the protected
   * SetCurrentPosition of the itkStandardGradientDescentOptimizer class.
   */
  void
  SetCurrentPositionPublic(const ParametersType & param) override
  {
    this->Superclass1::SetCurrentPosition(param);
  }


  /** Set the MaximumNumberOfSamplingAttempts. */
  itkSetMacro(MaximumNumberOfSamplingAttempts, unsigned long);

  /** Get the MaximumNumberOfSamplingAttempts. */
  itkGetConstReferenceMacro(MaximumNumberOfSamplingAttempts, unsigned long);

protected:
  StandardGradientDescent();
  ~StandardGradientDescent() override = default;

private:
  elxOverrideGetSelfMacro;

  /** Private variables for the sampling attempts. */
  unsigned long m_MaximumNumberOfSamplingAttempts;
  unsigned long m_CurrentNumberOfSamplingAttempts;
  unsigned long m_PreviousErrorAtIteration;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxStandardGradientDescent.hxx"
#endif

#endif // end #ifndef elxStandardGradientDescent_h
