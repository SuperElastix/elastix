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
#ifndef elxSimplex_h
#define elxSimplex_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkAmoebaOptimizer.h"

namespace elastix
{

/**
 * \class Simplex
 * \brief An optimizer based on Simplex...
 *
 * This optimizer is a wrap around the itk::AmoebaOptimizer.
 * This wrap-around class takes care of setting parameters, and printing progress
 * information.
 * For detailed information about the optimisation method, please read the
 * documentation of the itkAmoebaOptimizer (in the ITK-manual).
 * \sa ImprovedSimplexOptimizer
 * \ingroup Optimizers
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT Simplex
  : public itk::AmoebaOptimizer
  , public OptimizerBase<TElastix>
{
public:
  /** Standard ITK.*/
  typedef Simplex                       Self;
  typedef AmoebaOptimizer               Superclass1;
  typedef OptimizerBase<TElastix>       Superclass2;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(Simplex, AmoebaOptimizer);

  /** Name of this class.
   * Use this name in the parameter file to select this specific optimizer. \n
   * example: <tt>(Optimizer "Simplex")</tt>\n
   */
  elxClassNameMacro("Simplex");

  /** Typedef's inherited from Superclass1.*/
  typedef Superclass1::CostFunctionType    CostFunctionType;
  typedef Superclass1::CostFunctionPointer CostFunctionPointer;

  /** Typedef's inherited from Elastix.*/
  typedef typename Superclass2::ElastixType          ElastixType;
  typedef typename Superclass2::ElastixPointer       ElastixPointer;
  typedef typename Superclass2::ConfigurationType    ConfigurationType;
  typedef typename Superclass2::ConfigurationPointer ConfigurationPointer;
  typedef typename Superclass2::RegistrationType     RegistrationType;
  typedef typename Superclass2::RegistrationPointer  RegistrationPointer;
  typedef typename Superclass2::ITKBaseType          ITKBaseType;

  /** Typedef for the ParametersType. */
  typedef typename Superclass1::ParametersType ParametersType;

  /** Methods invoked by elastix, in which parameters can be set and
   * progress information can be printed. */
  void
  BeforeRegistration(void) override;

  void
  BeforeEachResolution(void) override;

  void
  AfterEachResolution(void) override;

  void
  AfterEachIteration(void) override;

  void
  AfterRegistration(void) override;

  /** Override the SetInitialPosition.
   * Override the implementation in itkOptimizer.h, to
   * ensure that the scales array and the parameters
   * array have the same size. */
  void
  SetInitialPosition(const ParametersType & param) override;

protected:
  Simplex() = default;
  ~Simplex() override = default;

private:
  elxOverrideGetSelfMacro;

  Simplex(const Self &) = delete;
  void
  operator=(const Self &) = delete;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxSimplex.hxx"
#endif

#endif // end #ifndef elxSimplex_h
