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
#ifndef itkDeformationVectorFieldTransform_hxx
#define itkDeformationVectorFieldTransform_hxx

#include "itkDeformationVectorFieldTransform.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#include "itkComposeImageFilter.h"

namespace itk
{

/**
 * *********************** Constructor **************************
 */

template <class TScalarType, unsigned int NDimensions>
DeformationVectorFieldTransform<TScalarType, NDimensions>::DeformationVectorFieldTransform()
{
  /** Initialize m_Images. */
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    this->m_Images[i] = nullptr;
  }

} // end Constructor


/**
 * *********************** Destructor ***************************
 */

template <class TScalarType, unsigned int NDimensions>
DeformationVectorFieldTransform<TScalarType, NDimensions>::~DeformationVectorFieldTransform()
{
  /** Initialize m_Images. */
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    this->m_Images[i] = nullptr;
  }
} // end Destructor


/**
 * ******************* SetCoefficientVectorImage **********************
 *
 * Convert VectorImage (= deformation field) to series of images.
 * Set the B-Spline coefficients using a deformation field
 * image as input.
 */

template <class TScalarType, unsigned int NDimensions>
void
DeformationVectorFieldTransform<TScalarType, NDimensions>::SetCoefficientVectorImage(
  const CoefficientVectorImageType * vecImage)
{
  /** Typedef's for iterators. */
  using VectorIteratorType = ImageRegionConstIterator<CoefficientVectorImageType>;
  using IteratorType = ImageRegionIterator<CoefficientImageType>;

  /** Create array of images representing the B-spline
   * coefficients in each dimension.
   */
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    this->m_Images[i] = CoefficientImageType::New();
    this->m_Images[i]->SetRegions(vecImage->GetLargestPossibleRegion());
    this->m_Images[i]->SetOrigin(vecImage->GetOrigin());
    this->m_Images[i]->SetSpacing(vecImage->GetSpacing());
    this->m_Images[i]->Allocate();
  }

  /** Setup the iterators. */
  VectorIteratorType vecit(vecImage, vecImage->GetLargestPossibleRegion());
  vecit.GoToBegin();
  IteratorType it[SpaceDimension];
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    it[i] = IteratorType(this->m_Images[i], this->m_Images[i]->GetLargestPossibleRegion());
    it[i].GoToBegin();
  }

  /** Copy one element of a vector to an image. */
  CoefficientVectorPixelType vect;
  while (!vecit.IsAtEnd())
  {
    vect = vecit.Get();
    for (unsigned int i = 0; i < SpaceDimension; ++i)
    {
      it[i].Set(static_cast<CoefficientPixelType>(vect[i]));
      ++it[i];
    }
    ++vecit;
  }

  /** Put it in the Superclass. */
  this->SetCoefficientImages(this->m_Images);

} // end SetCoefficientVectorImage()


/**
 * ******************* GetCoefficientVectorImage **********************
 *
 * Convert series of coefficient images to VectorImage (= deformation field).
 *
 */

template <class TScalarType, unsigned int NDimensions>
void
DeformationVectorFieldTransform<TScalarType, NDimensions>::GetCoefficientVectorImage(
  CoefficientVectorImagePointer & vecImage) const
{
  /** Typedef for the combiner. */
  using ScalarImageCombineType = ComposeImageFilter<CoefficientImageType, CoefficientVectorImageType>;

  /** Get a handle to the series of coefficient images. */
  const CoefficientImagePointer * coefImage = this->GetCoefficientImages();

  /** Combine the coefficient images to a vector image. */
  auto combiner = ScalarImageCombineType::New();
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    combiner->SetInput(i, coefImage[i]);
  }
  vecImage = combiner->GetOutput();
  vecImage->Update();

} // end GetCoefficientVectorImage()


} // end namespace itk

#endif // end #ifndef itkDeformationVectorFieldTransform_hxx
