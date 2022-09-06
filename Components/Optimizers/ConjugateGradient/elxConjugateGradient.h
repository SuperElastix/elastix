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
#ifndef elxConjugateGradient_h
#define elxConjugateGradient_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkGenericConjugateGradientOptimizer.h"
#include "itkMoreThuenteLineSearchOptimizer.h"

namespace elastix
{

/**
 * \class ConjugateGradient
 * \brief An optimizer based on the itk::GenericConjugateGradientOptimizer.
 *
 * A ConjugateGradient optimizer, using the itk::MoreThuenteLineSearchOptimizer.
 * Different conjugate gradient methods can be selected with this optimizer.
 *
 * This optimizer support the NewSamplesEveryIteration option. It requests
 * new samples for the computation of each search direction (not during
 * the line search). Actually this makes no sense for a conjugate gradient optimizer.
 * So, think twice before using the NewSamplesEveryIteration option.
 *
 * The parameters used in this class are:
 * \parameter Optimizer: Select this optimizer as follows:\n
 *    <tt>(Optimizer "ConjugateGradient")</tt>
 * \parameter GenerateLineSearchIterations: Whether line search iteration
 *   should be counted as elastix-iterations.\n
 *   example: <tt>(GenerateLineSearchIterations "true")</tt>\n
 *   Can only be specified for all resolutions at once. \n
 *   Default value: "false".\n
 * \parameter MaximumNumberOfIterations: The maximum number of iterations in each resolution. \n
 *    example: <tt>(MaximumNumberOfIterations 100 100 50)</tt> \n
 *    Default value: 100.\n
 * \parameter MaximumNumberOfLineSearchIterations: The maximum number of iterations in each resolution. \n
 *    example: <tt>(MaximumNumberOfIterations 10 10 5)</tt> \n
 *    Default value: 10.\n
 * \parameter StepLength: Set the length of the initial step tried by the
 *    itk::MoreThuenteLineSearchOptimizer.\n
 *    example: <tt>(StepLength 2.0 1.0 0.5)</tt> \n
 *    Default value: 1.0.\n
 * \parameter LineSearchValueTolerance: Determine the Wolfe conditions that the
 *    itk::MoreThuenteLineSearchOptimizer tries to satisfy.\n
 *    example: <tt>(LineSearchValueTolerance 0.0001 0.0001 0.0001)</tt> \n
 *    Default value: 0.0001.\n
 * \parameter LineSearchGradientTolerance: Determine the Wolfe conditions that the
 *    itk::MoreThuenteLineSearchOptimizer tries to satisfy.\n
 *    example: <tt>(LineSearchGradientTolerance 0.9 0.9 0.9)</tt> \n
 *    Default value: 0.9.\n
 * \parameter ValueTolerance: Stopping criterion. See the documentation of the
 *    itk::GenericConjugateGradientOptimizer for more information.\n
 *    example: <tt>(ValueTolerance 0.001 0.0001 0.000001)</tt> \n
 *    Default value: 0.00001.\n
 * \parameter GradientMagnitudeTolerance: Stopping criterion. See the documentation of the
 *    itk::GenericConjugateGradientOptimizer for more information.\n
 *    example: <tt>(GradientMagnitudeTolerance 0.001 0.0001 0.000001)</tt> \n
 *    Default value: 0.000001.\n
 * \parameter ConjugateGradientType: a string that defines how 'beta' is computed in each resolution.
 *    The following methods are implemented: "SteepestDescent", "FletcherReeves", "PolakRibiere",
 *    "DaiYuan", "HestenesStiefel", and "DaiYuanHestenesStiefel". "SteepestDescent" simply sets beta=0.
 *    See the source code of the GenericConjugateGradientOptimizer for more information.\n
 *    example: <tt>(ConjugateGradientType "FletcherReeves" "PolakRibiere")</tt> \n
 *    Default value: "DaiYuanHestenesStiefel".\n
 * \parameter StopIfWolfeNotSatisfied: Whether to stop the optimisation if in one iteration
 *    the Wolfe conditions can not be satisfied by the itk::MoreThuenteLineSearchOptimizer.\n
 *    In general it is wise to do so.\n
 *    example: <tt>(StopIfWolfeNotSatisfied "true" "false")</tt> \n
 *    Default value: "true".\n
 *
 *
 * \ingroup Optimizers
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT ConjugateGradient
  : public itk::GenericConjugateGradientOptimizer
  , public OptimizerBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ConjugateGradient);

  /** Standard ITK.*/
  using Self = ConjugateGradient;
  using Superclass1 = GenericConjugateGradientOptimizer;
  using Superclass2 = OptimizerBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ConjugateGradient, GenericConjugateGradientOptimizer);

  /** Name of this class.
   * Use this name in the parameter file to select this specific optimizer. \n
   * example: <tt>(Optimizer "ConjugateGradient")</tt>\n
   */
  elxClassNameMacro("ConjugateGradient");

  /** Typedef's inherited from Superclass1.*/
  using Superclass1::CostFunctionType;
  using Superclass1::CostFunctionPointer;
  using Superclass1::StopConditionType;
  using Superclass1::ParametersType;
  using Superclass1::DerivativeType;
  using Superclass1::ScalesType;

  /** Typedef's inherited from Elastix.*/
  using typename Superclass2::ElastixType;
  using typename Superclass2::RegistrationType;
  using ITKBaseType = typename Superclass2::ITKBaseType;

  /** Extra typedefs */
  using LineOptimizerType = itk::MoreThuenteLineSearchOptimizer;
  using LineOptimizerPointer = LineOptimizerType::Pointer;
  using EventPassThroughType = itk::ReceptorMemberCommand<Self>;
  using EventPassThroughPointer = typename EventPassThroughType::Pointer;

  /** Check if any scales are set, and set the UseScales flag on or off;
   * after that call the superclass' implementation */
  void
  StartOptimization() override;

  /** Methods to set parameters and print output at different stages
   * in the registration process.*/
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

  itkGetConstMacro(StartLineSearch, bool);

protected:
  ConjugateGradient();
  ~ConjugateGradient() override = default;

  LineOptimizerPointer m_LineOptimizer;

  /** Convert the line search stop condition to a string */
  virtual std::string
  GetLineSearchStopCondition() const;

  /** Generate a string, representing the phase of optimisation
   * (line search, main) */
  virtual std::string
  DeterminePhase() const;

  /** Reimplement the superclass. Calls the superclass' implementation
   * and checks if the MoreThuente line search routine has stopped with
   * Wolfe conditions satisfied. */
  bool
  TestConvergence(bool firstLineSearchDone) override;

  /** Call the superclass' implementation. If an itk::ExceptionObject is caught,
   * because the line search optimizer tried a too big step, the exception
   * is printed, but ignored further. The optimizer stops, but elastix
   * just goes on to the next resolution. */
  void
  LineSearch(const ParametersType searchDir, double & step, ParametersType & x, MeasureType & f, DerivativeType & g)
    override;

private:
  elxOverrideGetSelfMacro;

  void
  InvokeIterationEvent(const itk::EventObject & event);

  EventPassThroughPointer m_EventPasser;
  double                  m_SearchDirectionMagnitude;
  bool                    m_StartLineSearch;
  bool                    m_GenerateLineSearchIterations;
  bool                    m_StopIfWolfeNotSatisfied;
  bool                    m_WolfeIsStopCondition;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxConjugateGradient.hxx"
#endif

#endif // end #ifndef elxConjugateGradient_h
