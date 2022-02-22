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
#ifndef itkBSplineInterpolationWeightFunctionBase_hxx
#define itkBSplineInterpolationWeightFunctionBase_hxx

#include "itkBSplineInterpolationWeightFunctionBase.h"
#include "itkImage.h"
#include "itkMatrix.h"
#include "itkImageRegionConstIteratorWithIndex.h"

namespace itk
{

/**
 * ****************** Constructor *******************************
 */

template <class TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder>
BSplineInterpolationWeightFunctionBase<TCoordRep, VSpaceDimension, VSplineOrder>::
  BSplineInterpolationWeightFunctionBase()
{
  /** Initialize members. */
  this->InitializeSupport();
  this->InitializeOffsetToIndexTable();

} // end Constructor


/**
 * ******************* InitializeSupport *******************
 */

template <class TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder>
void
BSplineInterpolationWeightFunctionBase<TCoordRep, VSpaceDimension, VSplineOrder>::InitializeSupport()
{
  /** Initialize support region. */
  this->m_SupportSize.Fill(SplineOrder + 1);

  /** Initialize the number of weights. */
  this->m_NumberOfWeights = 1;
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    this->m_NumberOfWeights *= this->m_SupportSize[i];
  }

} // end InitializeSupport()


/**
 * ******************* InitializeOffsetToIndexTable *******************
 */

template <class TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder>
void
BSplineInterpolationWeightFunctionBase<TCoordRep, VSpaceDimension, VSplineOrder>::InitializeOffsetToIndexTable()
{
  /** Create a temporary image. */
  using CharImageType = Image<char, SpaceDimension>;
  auto tempImage = CharImageType::New();
  tempImage->SetRegions(this->m_SupportSize);
  tempImage->Allocate();

  /** Create an iterator over the image. */
  using IteratorType = ImageRegionConstIteratorWithIndex<CharImageType>;
  IteratorType it(tempImage, tempImage->GetBufferedRegion());
  it.GoToBegin();

  /** Fill the OffsetToIndexTable. */
  this->m_OffsetToIndexTable.set_size(this->m_NumberOfWeights, SpaceDimension);
  unsigned long counter = 0;
  while (!it.IsAtEnd())
  {
    IndexType ind = it.GetIndex();
    for (unsigned int i = 0; i < SpaceDimension; ++i)
    {
      this->m_OffsetToIndexTable[counter][i] = ind[i];
    }

    ++counter;
    ++it;
  }

} // end InitializeOffsetToIndexTable()


/**
 * ******************* PrintSelf *******************
 */

template <class TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder>
void
BSplineInterpolationWeightFunctionBase<TCoordRep, VSpaceDimension, VSplineOrder>::PrintSelf(std::ostream & os,
                                                                                            Indent         indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "NumberOfWeights: " << this->m_NumberOfWeights << std::endl;
  os << indent << "SupportSize: " << this->m_SupportSize << std::endl;
  os << indent << "OffsetToIndexTable: " << this->m_OffsetToIndexTable << std::endl;

} // end PrintSelf()


/**
 * ******************* ComputeStartIndex *******************
 */

template <class TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder>
void
BSplineInterpolationWeightFunctionBase<TCoordRep, VSpaceDimension, VSplineOrder>::ComputeStartIndex(
  const ContinuousIndexType & cindex,
  IndexType &                 startIndex) const
{
  /** Find the starting index of the support region. */
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    startIndex[i] = static_cast<typename IndexType::IndexValueType>(
      std::floor(cindex[i] - static_cast<double>(this->m_SupportSize[i] - 2.0) / 2.0));
  }

} // end ComputeStartIndex()


/**
 * ******************* Evaluate *******************
 */

template <class TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder>
auto
BSplineInterpolationWeightFunctionBase<TCoordRep, VSpaceDimension, VSplineOrder>::Evaluate(
  const ContinuousIndexType & cindex) const -> WeightsType
{
  /** Construct arguments for the Evaluate function that really does the work. */
  WeightsType weights;
  IndexType   startIndex;
  this->ComputeStartIndex(cindex, startIndex);

  /** Call the Evaluate function that really does the work. */
  this->Evaluate(cindex, startIndex, weights);

  return weights;

} // end Evaluate()


/**
 * ******************* Evaluate *******************
 */

template <class TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder>
void
BSplineInterpolationWeightFunctionBase<TCoordRep, VSpaceDimension, VSplineOrder>::Evaluate(
  const ContinuousIndexType & cindex,
  const IndexType &           startIndex,
  WeightsType &               weights) const
{
  /** Don't initialize the weights!
   * weights.SetSize( this->m_NumberOfWeights );
   * This will result in a big performance penalty (50%). In Evaluate( index )
   * we have set the size correctly anyway. We just assume that when this
   * function is called directly, the user has set the size correctly.
   */

  /** Compute the 1D weights. */
  OneDWeightsType weights1D;
  this->Compute1DWeights(cindex, startIndex, weights1D);

  /** Compute the vector of weights. */
  for (unsigned int k = 0; k < this->m_NumberOfWeights; ++k)
  {
    double                tmp1 = 1.0;
    const unsigned long * tmp2 = this->m_OffsetToIndexTable[k];
    for (unsigned int j = 0; j < SpaceDimension; ++j)
    {
      tmp1 *= weights1D[j][tmp2[j]];
    }
    weights[k] = tmp1;
  }

} // end Evaluate()


} // end namespace itk

#endif
