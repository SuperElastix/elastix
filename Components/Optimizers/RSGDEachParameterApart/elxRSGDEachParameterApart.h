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
#ifndef elxRSGDEachParameterApart_h
#define elxRSGDEachParameterApart_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkRSGDEachParameterApartOptimizer.h"

namespace elastix
{

/**
 * \class RSGDEachParameterApart
 * \brief An optimizer based on gradient descent.
 *
 * The underlying itk class is almost a copy of the normal
 * RegularStepGradientDescent. The difference is that each
 * parameter has its own step length, whereas the normal
 * RSGD has one step length that is used for all parameters.
 *
 * This could cause inaccuracies, if, for example, parameter
 * 1, 2 and 3 are already close to the optimum, but parameter
 * 4 not yet. The average stepsize is halved then, so parameter
 * 4 will not have time to reach its optimum (in a worst case
 * scenario).
 *
 * The RSGDEachParameterApart stops only if ALL steplenghts
 * are smaller than the MinimumStepSize given in the parameter
 * file!
 *
 * The elastix shell class (so, this class...), is a copy of
 * the elxRegularStepGradientDescent, so the parameters in the
 * parameter file, the output etc are similar.
 *
 * The parameters used in this class are:
 * \parameter Optimizer: Select this optimizer as follows:\n
 *    <tt>(Optimizer "RSGDEachParameterApart")</tt>
 * \parameter MaximumNumberOfIterations: the maximum number of iterations in each resolution. \n
 *   example: <tt>(MaximumNumberOfIterations 100 100 50)</tt> \n
 *   Default value: 100.
 * \parameter MinimumGradientMagnitude: stopping criterion. If the magnitude of the derivative
 *   of the cost function is below this value, optimisation is stopped. \n
 *   example: <tt>(MinimumGradientMagnitude 0.0001 0.0001 0.001)</tt> \n
 *   Default value: 1e-8.
 * \parameter MinimumStepLength: stopping criterion. If the steplength is below this
 *   value, optimisation is stopped. \n
 *   example: <tt>(MinimumStepLength 1.0 0.5 0.1)</tt> \n
 *   Default value: <em>0.5 / 2^resolutionlevel</em>
 * \parameter MaximumStepLength: the starting steplength.  \n
 *   example: <tt>(MaxiumStepLength 16.0 8.0 4.0)</tt> \n
 *   Default value: <em>16 / 2^resolutionlevel</em>.
 *
 * \ingroup Optimizers
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT RSGDEachParameterApart
  : public itk::RSGDEachParameterApartOptimizer
  , public OptimizerBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(RSGDEachParameterApart);

  /** Standard ITK.*/
  using Self = RSGDEachParameterApart;
  using Superclass1 = RSGDEachParameterApartOptimizer;
  using Superclass2 = OptimizerBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(RSGDEachParameterApart, RSGDEachParameterApartOptimizer);

  /** Name of this class.
   * Use this name in the parameter file to select this specific optimizer. \n
   * example: <tt>(Optimizer "RSGDEachParameterApart")</tt>\n
   */
  elxClassNameMacro("RSGDEachParameterApart");

  /** Typedef's inherited from Superclass1.*/
  using Superclass1::CostFunctionType;
  using Superclass1::CostFunctionPointer;

  /** Typedef's inherited from Elastix.*/
  using typename Superclass2::ElastixType;
  using typename Superclass2::RegistrationType;
  using ITKBaseType = typename Superclass2::ITKBaseType;

  /** Typedef for the ParametersType. */
  using typename Superclass1::ParametersType;

  /** Methods that have to be present everywhere.*/
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
  RSGDEachParameterApart() = default;
  ~RSGDEachParameterApart() override = default;

private:
  elxOverrideGetSelfMacro;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxRSGDEachParameterApart.hxx"
#endif

#endif // end #ifndef elxRSGDEachParameterApart_h
