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

#include "itkRSGDEachParameterApartOptimizer.h"
#include "itkCommand.h"
#include "itkEventObject.h"

namespace itk
{

/**
 * Advance one Step following the gradient direction
 * This method will be overrided in non-vector spaces
 */
void
RSGDEachParameterApartOptimizer::StepAlongGradient(const DerivativeType & factor,
                                                   const DerivativeType & transformedGradient)
{

  itkDebugMacro("factor = " << factor << "  transformedGradient= " << transformedGradient);

  const unsigned int spaceDimension = m_CostFunction->GetNumberOfParameters();

  ParametersType newPosition(spaceDimension);
  ParametersType currentPosition = this->GetCurrentPosition();

  for (unsigned int j = 0; j < spaceDimension; ++j)
  {
    /** Each parameters has its own factor! */
    newPosition[j] = currentPosition[j] + transformedGradient[j] * factor[j];
  }

  itkDebugMacro("new position = " << newPosition);

  this->SetCurrentPosition(newPosition);
}


} // end namespace itk
