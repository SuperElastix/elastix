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
#ifndef itkAffineLogTransform_hxx
#define itkAffineLogTransform_hxx

#include <vnl/vnl_matrix_exp.h>
#include "itkMath.h"
#include "itkAffineLogTransform.h"

namespace itk
{

// Constructor with default arguments
template <class TScalarType, unsigned int Dimension>
AffineLogTransform<TScalarType, Dimension>::AffineLogTransform()
  : Superclass(ParametersDimension)
{
  this->m_MatrixLogDomain.Fill(itk::NumericTraits<ScalarType>::Zero);
  this->PrecomputeJacobianOfSpatialJacobian();
}


// Constructor with default arguments
template <class TScalarType, unsigned int Dimension>
AffineLogTransform<TScalarType, Dimension>::AffineLogTransform(const MatrixType &      matrix,
                                                               const OutputPointType & offset)
{
  this->SetMatrix(matrix);

  OffsetType off;
  for (unsigned int i = 0; i < Dimension; ++i)
  {
    off[i] = offset[i];
  }
  this->SetOffset(off);

  this->PrecomputeJacobianOfSpatialJacobian();
}


// Constructor with arguments
template <class TScalarType, unsigned int Dimension>
AffineLogTransform<TScalarType, Dimension>::AffineLogTransform(unsigned int spaceDimension,
                                                               unsigned int parametersDimension)
  : Superclass(spaceDimension, parametersDimension)
{
  this->m_MatrixLogDomain.Fill(itk::NumericTraits<ScalarType>::Zero);
  this->PrecomputeJacobianOfSpatialJacobian();
}


// Set Parameters
template <class TScalarType, unsigned int Dimension>
void
AffineLogTransform<TScalarType, Dimension>::SetParameters(const ParametersType & parameters)
{
  itkDebugMacro(<< "Setting parameters " << parameters);
  unsigned int k = 0; // Dummy loop index

  MatrixType exponentMatrix;

  for (unsigned int i = 0; i < Dimension; ++i)
  {
    for (unsigned int j = 0; j < Dimension; ++j)
    {
      this->m_MatrixLogDomain(i, j) = parameters[k];
      k += 1;
    }
  }

  exponentMatrix = vnl_matrix_exp(this->m_MatrixLogDomain.GetVnlMatrix());

  this->PrecomputeJacobianOfSpatialJacobian();

  this->SetVarMatrix(exponentMatrix);

  OutputVectorType off;

  for (unsigned int i = 0; i < Dimension; ++i)
  {
    off[i] = parameters[k];
    k += 1;
  }

  this->SetVarTranslation(off);
  this->ComputeOffset();

  // Modified is always called since we just have a pointer to the
  // parameters and cannot know if the parameters have changed.

  this->Modified();
  itkDebugMacro(<< "After setting parameters ");
}


// Get Parameters
template <class TScalarType, unsigned int Dimension>
auto
AffineLogTransform<TScalarType, Dimension>::GetParameters() const -> const ParametersType &
{
  unsigned int k = 0; // Dummy loop index

  for (unsigned int i = 0; i < Dimension; ++i)
  {
    for (unsigned int j = 0; j < Dimension; ++j)
    {
      this->m_Parameters[k] = this->m_MatrixLogDomain(i, j);
      k += 1;
    }
  }

  for (unsigned int j = 0; j < Dimension; ++j)
  {
    this->m_Parameters[k] = this->GetTranslation()[j];
    k += 1;
  }

  return this->m_Parameters;
}

// SetIdentity
template <class TScalarType, unsigned int Dimension>
void
AffineLogTransform<TScalarType, Dimension>::SetIdentity()
{
  Superclass::SetIdentity();
  this->m_MatrixLogDomain.Fill(itk::NumericTraits<ScalarType>::Zero);
  this->PrecomputeJacobianOfSpatialJacobian();
}


// Get Jacobian
template <class TScalarType, unsigned int Dimension>
void
AffineLogTransform<TScalarType, Dimension>::GetJacobian(const InputPointType &       p,
                                                        JacobianType &               j,
                                                        NonZeroJacobianIndicesType & nzji) const
{
  unsigned int d = Dimension;

  j.SetSize(d, ParametersDimension);
  j.Fill(itk::NumericTraits<ScalarType>::Zero);

  const JacobianOfSpatialJacobianType & jsj = this->m_JacobianOfSpatialJacobian;
  const InputVectorType                 pp = p - this->GetCenter();
  for (unsigned int dim = 0; dim < d * d; ++dim)
  {
    const InputVectorType column = jsj[dim] * pp;
    for (unsigned int i = 0; i < d; ++i)
    {
      j(i, dim) = column[i];
    }
  }

  // compute derivatives for the translation part
  const unsigned int blockOffset = d * d;
  for (unsigned int dim = 0; dim < Dimension; ++dim)
  {
    j[dim][blockOffset + dim] = 1.0;
  }

  nzji = this->m_NonZeroJacobianIndices;
}


// Precompute Jacobian of Spatial Jacobian
template <class TScalarType, unsigned int Dimension>
void
AffineLogTransform<TScalarType, Dimension>::PrecomputeJacobianOfSpatialJacobian()
{
  unsigned int d = Dimension;

  /** The Jacobian of spatial Jacobian is constant over inputspace, so is precomputed */
  JacobianOfSpatialJacobianType & jsj = this->m_JacobianOfSpatialJacobian;

  jsj.resize(ParametersDimension);

  vnl_matrix<ScalarType> dA(d, d);

  vnl_matrix<ScalarType> dummymatrix(d, d);

  vnl_matrix<ScalarType> A_bar(2 * d, 2 * d);

  vnl_matrix<ScalarType> B_bar(2 * d, 2 * d);

  dA.fill(itk::NumericTraits<ScalarType>::Zero);
  dummymatrix.fill(itk::NumericTraits<ScalarType>::Zero);
  A_bar.fill(itk::NumericTraits<ScalarType>::Zero);

  // Fill A_bar top left and bottom right with A
  for (unsigned int k = 0; k < d; ++k)
  {
    for (unsigned int l = 0; l < d; ++l)
    {
      A_bar(k, l) = this->m_MatrixLogDomain(k, l);
    }
  }
  for (unsigned int k = d; k < 2 * d; ++k)
  {
    for (unsigned int l = d; l < 2 * d; ++l)
    {
      A_bar(k, l) = this->m_MatrixLogDomain(k - d, l - d);
    }
  }

  unsigned int m = 0; // Dummy loop index

  // Non-translation derivatives
  for (unsigned int i = 0; i < d; ++i)
  {
    for (unsigned int j = 0; j < d; ++j)
    {
      dA(i, j) = 1;
      for (unsigned int k = 0; k < d; ++k)
      {
        for (unsigned int l = d; l < 2 * d; ++l)
        {
          A_bar(k, l) = dA(k, (l - d));
        }
      }
      B_bar = vnl_matrix_exp(A_bar);
      for (unsigned int k = 0; k < d; ++k)
      {
        for (unsigned int l = d; l < 2 * d; ++l)
        {
          dummymatrix(k, (l - d)) = B_bar(k, l);
        }
      }
      jsj[m] = dummymatrix;
      dA.fill(itk::NumericTraits<ScalarType>::Zero);
      m += 1;
    }
  }

  /** Translation parameters: */
  for (unsigned int par = d * d; par < ParametersDimension; ++par)
  {
    jsj[par].Fill(itk::NumericTraits<ScalarType>::Zero);
  }
}


// Print self
template <class TScalarType, unsigned int Dimension>
void
AffineLogTransform<TScalarType, Dimension>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "parameters:" << this->m_Parameters << std::endl;
}


} // namespace itk

#endif // itkAffineLogTransform_hxx
