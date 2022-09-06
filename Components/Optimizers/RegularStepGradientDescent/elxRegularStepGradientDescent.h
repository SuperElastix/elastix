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
#ifndef elxRegularStepGradientDescent_h
#define elxRegularStepGradientDescent_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkRegularStepGradientDescentOptimizer.h"

namespace elastix
{

/**
 * \class RegularStepGradientDescent
 * \brief An optimizer based on gradient descent...
 *
 * This optimizer is a wrap around the itk::RegularStepGradientDescentOptimizer.
 * This wrap-around class takes care of setting parameters, and printing progress
 * information.
 * For detailed information about the optimisation method, please read the
 * documentation of the itkRegularStepGradientDescentOptimizer (in the ITK-manual).
 *
 * The parameters used in this class are:
 * \parameter Optimizer: Select this optimizer as follows:\n
 *    <tt>(Optimizer "RegularStepGradientDescent")</tt>
 * \parameter MaximumNumberOfIterations: the maximum number of iterations in each resolution. \n
 *   example: <tt>(MaximumNumberOfIterations 100 100 50)</tt> \n
 *   Default value: 500.
 * \parameter MinimumGradientMagnitude: stopping criterion. If the magnitude of the derivative
 *   of the cost function is below this value, optimisation is stopped. \n
 *   example: <tt>(MinimumGradientMagnitude 0.0001 0.0001 0.001)</tt> \n
 *   Default value: 1e-8.
 * \parameter MinimumStepLength: stopping criterion. If the steplength is below this
 *   value, optimisation is stopped. \n
 *   example: <tt>(MinimumStepLength 1.0 0.5 0.1)</tt> \n
 *   Default value: <em>0.5 / 2^resolutionlevel</em>
 * \parameter MaximumStepLength: the starting steplength.  \n
 *   example: <tt>(MaximumStepLength 16.0 8.0 4.0)</tt> \n
 *   Default value: <em>16 / 2^resolutionlevel</em>.
 * \parameter RelaxationFactor: the factor with which the steplength is multiplied,
 *   if the optimiser notices that a smaller steplength is needed. \n
 *   example: <tt>(RelaxationFactor 0.5 0.5 0.8 )</tt>. \n
 *   Default/recommended value: 0.5.
 *
 *
 * \sa ImprovedRegularStepGradientDescentOptimizer
 * \ingroup Optimizers
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT RegularStepGradientDescent
  : public itk::RegularStepGradientDescentOptimizer
  , public OptimizerBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(RegularStepGradientDescent);

  /** Standard ITK.*/
  using Self = RegularStepGradientDescent;
  using Superclass1 = RegularStepGradientDescentOptimizer;
  using Superclass2 = OptimizerBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(RegularStepGradientDescent, RegularStepGradientDescentOptimizer);

  /** Name of this class.
   * Use this name in the parameter file to select this specific optimizer. \n
   * example: <tt>(Optimizer "RegularStepGradientDescent")</tt>\n
   */
  elxClassNameMacro("RegularStepGradientDescent");

  /** Typedef's inherited from Superclass1.*/
  using Superclass1::CostFunctionType;
  using Superclass1::CostFunctionPointer;

  /** Typedef's inherited from Elastix.*/
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

  /** Override the SetInitialPosition.
   * Override the implementation in itkOptimizer.h, to
   * ensure that the scales array and the parameters
   * array have the same size. */
  void
  SetInitialPosition(const ParametersType & param) override;

protected:
  RegularStepGradientDescent() = default;
  ~RegularStepGradientDescent() override = default;

private:
  elxOverrideGetSelfMacro;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxRegularStepGradientDescent.hxx"
#endif

#endif // end #ifndef elxRegularStepGradientDescent_h
