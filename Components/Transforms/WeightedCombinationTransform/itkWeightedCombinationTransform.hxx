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
#ifndef _itkWeightedCombinationTransform_hxx
#define _itkWeightedCombinationTransform_hxx

#include "itkWeightedCombinationTransform.h"

namespace itk
{

/**
 * ********************* Constructor ****************************
 */

template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
WeightedCombinationTransform<TScalarType, NInputDimensions, NOutputDimensions>::WeightedCombinationTransform()
  : Superclass(OutputSpaceDimension)
{
  this->m_SumOfWeights = 1.0;
  this->m_NormalizeWeights = false;
  this->m_HasNonZeroSpatialHessian = true;
  this->m_HasNonZeroJacobianOfSpatialHessian = true;
} // end Constructor


/**
 * ************************ SetParameters ***********************
 */

template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
void
WeightedCombinationTransform<TScalarType, NInputDimensions, NOutputDimensions>::SetParameters(
  const ParametersType & param)
{
  if (param.GetSize() != this->m_TransformContainer.size())
  {
    itkExceptionMacro(
      << "Number of parameters does not match the number of transforms set in the transform container.");
  }

  // this->m_Jacobian.SetSize( OutputSpaceDimension, param.GetSize() );
  this->m_Parameters = param;
  this->m_SumOfWeights = param.sum();
  if (this->m_SumOfWeights < 1e-10 && this->m_NormalizeWeights)
  {
    itkExceptionMacro(<< "Sum of weights for WeightedCombinationTransform is smaller than 0.");
  }

  // Precompute the nonzerojacobianindices vector
  const NumberOfParametersType nrParams = param.GetSize();
  if (nrParams != this->m_NonZeroJacobianIndices.size())
  {
    this->m_NonZeroJacobianIndices.resize(nrParams);
    for (unsigned int i = 0; i < nrParams; ++i)
    {
      this->m_NonZeroJacobianIndices[i] = i;
    }
  }

  this->Modified();

} // end SetParameters()


/**
 * ********************* TransformPoint ****************************
 */

template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
auto
WeightedCombinationTransform<TScalarType, NInputDimensions, NOutputDimensions>::TransformPoint(
  const InputPointType & inputPoint) const -> OutputPointType
{
  OutputPointType opp;
  opp.Fill(0.0);
  OutputPointType                tempopp;
  const TransformContainerType & tc = this->m_TransformContainer;
  const unsigned int             N = tc.size();
  const ParametersType &         param = this->m_Parameters;

  /** Calculate sum_i w_i T_i(x) */
  for (unsigned int i = 0; i < N; ++i)
  {
    tempopp = tc[i]->TransformPoint(inputPoint);
    const double w = param[i];
    for (unsigned int d = 0; d < OutputSpaceDimension; ++d)
    {
      opp[d] += w * tempopp[d];
    }
  }

  if (this->m_NormalizeWeights)
  {
    /** T(x) = \sum_i w_i T_i(x) / sum_i w_i */
    for (unsigned int d = 0; d < OutputSpaceDimension; ++d)
    {
      opp[d] /= this->m_SumOfWeights;
    }
  }
  else
  {
    /** T(x) = (1 - \sum_i w_i ) x + \sum_i w_i T_i(x) */
    const double factor = 1.0 - this->m_SumOfWeights;
    for (unsigned int d = 0; d < OutputSpaceDimension; ++d)
    {
      opp[d] += factor * inputPoint[d];
    }
  }
  return opp;

} // end TransformPoint


/**
 * ********************* GetJacobian ****************************
 */

template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
void
WeightedCombinationTransform<TScalarType, NInputDimensions, NOutputDimensions>::GetJacobian(
  const InputPointType &       inputPoint,
  JacobianType &               jac,
  NonZeroJacobianIndicesType & nzji) const
{
  OutputPointType                tempopp;
  const TransformContainerType & tc = this->m_TransformContainer;
  const unsigned int             N = tc.size();
  const ParametersType &         param = this->m_Parameters;
  jac.SetSize(OutputSpaceDimension, N);

  /** This transform has only nonzero jacobians. */
  nzji = this->m_NonZeroJacobianIndices;

  if (this->m_NormalizeWeights)
  {
    /** dT/dmu_i = ( T_i(x) - T(x) ) / ( \sum_i w_i ) */
    OutputPointType opp;
    opp.Fill(0.0);
    for (unsigned int i = 0; i < N; ++i)
    {
      tempopp = tc[i]->TransformPoint(inputPoint);
      const double w = param[i];
      for (unsigned int d = 0; d < OutputSpaceDimension; ++d)
      {
        opp[d] += w * tempopp[d];
        jac(d, i) = tempopp[d];
      }
    }
    for (unsigned int d = 0; d < OutputSpaceDimension; ++d)
    {
      opp[d] /= this->m_SumOfWeights;
    }
    for (unsigned int i = 0; i < N; ++i)
    {
      for (unsigned int d = 0; d < OutputSpaceDimension; ++d)
      {
        jac(d, i) = (jac(d, i) - opp[d]) / this->m_SumOfWeights;
      }
    }
  }
  else
  {
    /** dT/dmu_i = T_i(x) - x */
    for (unsigned int i = 0; i < N; ++i)
    {
      tempopp = tc[i]->TransformPoint(inputPoint);
      for (unsigned int d = 0; d < OutputSpaceDimension; ++d)
      {
        jac(d, i) = tempopp[d] - inputPoint[d];
      }
    }
  }

} // end GetJacobian()


} // end namespace itk

#endif
