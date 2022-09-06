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

#ifndef itkRSGDEachParameterApartOptimizer_h
#define itkRSGDEachParameterApartOptimizer_h

#include "itkRSGDEachParameterApartBaseOptimizer.h"

namespace itk
{

/**
 * \class RSGDEachParameterApartOptimizer
 * \brief An optimizer based on gradient descent.
 *
 * This class is almost a copy of the normal
 * itk::RegularStepGradientDescentOptimizer. The difference
 * is that each parameter has its own step length, whereas the normal
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
 * Note that this is a quite experimental optimizer, currently
 * only used for some specific tests.
 *
 * \ingroup Optimizers
 * \sa RSGDEachParameterApart
 */

class RSGDEachParameterApartOptimizer : public RSGDEachParameterApartBaseOptimizer
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(RSGDEachParameterApartOptimizer);

  /** Standard class typedefs. */
  using Self = RSGDEachParameterApartOptimizer;
  using Superclass = RSGDEachParameterApartBaseOptimizer;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(RSGDEachParameterApartOptimizer, RSGDEachParameterApartBaseOptimizer);

  /** Cost function typedefs. */
  using Superclass::CostFunctionType;
  using CostFunctionPointer = CostFunctionType::Pointer;

protected:
  RSGDEachParameterApartOptimizer() = default;
  ~RSGDEachParameterApartOptimizer() override = default;

  /** Advance one step along the corrected gradient taking into
   * account the steplengths represented by the factor array.
   * This method is invoked by AdvanceOneStep. It is expected
   * to be overrided by optimization methods in non-vector spaces
   * \sa AdvanceOneStep */
  void
  StepAlongGradient(const DerivativeType & factor, const DerivativeType & transformedGradient) override;
};

} // end namespace itk

#endif // end #ifndef itkRSGDEachParameterApartOptimizer_h
