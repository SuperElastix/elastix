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

#ifndef itkDeformationFieldRegulizer_hxx
#define itkDeformationFieldRegulizer_hxx

#include "itkDeformationFieldRegulizer.h"

namespace itk
{

/**
 * ************************ Constructor *************************
 */

template <class TAnyITKTransform>
DeformationFieldRegulizer<TAnyITKTransform>::DeformationFieldRegulizer()
{
  /** Initialize. */
  this->m_IntermediaryDeformationFieldTransform = nullptr;
  this->m_Initialized = false;

} // end Constructor


/**
 * ********* InitializeIntermediaryDeformationField **************
 */

template <class TAnyITKTransform>
void
DeformationFieldRegulizer<TAnyITKTransform>::InitializeDeformationFields()
{
  /** Initialize this->m_IntermediaryDeformationFieldTransform. */
  this->m_IntermediaryDeformationFieldTransform = IntermediaryDFTransformType::New();

  /** Initialize this->m_IntermediaryDeformationField. */
  auto intermediaryDeformationField = VectorImageType::New();
  intermediaryDeformationField->SetRegions(this->m_DeformationFieldRegion);
  intermediaryDeformationField->SetSpacing(this->m_DeformationFieldSpacing);
  intermediaryDeformationField->SetOrigin(this->m_DeformationFieldOrigin);
  try
  {
    intermediaryDeformationField->Allocate();
  }
  catch (itk::ExceptionObject & excp)
  {
    /** Add information to the exception and throw again. */
    excp.SetLocation("DeformationFieldRegulizer - InitializeDeformationFields()");
    std::string err_str = excp.GetDescription();
    err_str += "\nError occurred while allocating the intermediary deformation field.\n";
    excp.SetDescription(err_str);
    throw;
  }

  /** Set everything to zero. */
  IteratorType    it(intermediaryDeformationField, intermediaryDeformationField->GetLargestPossibleRegion());
  VectorPixelType vec;
  vec.Fill(NumericTraits<ScalarType>::ZeroValue());
  while (!it.IsAtEnd())
  {
    it.Set(vec);
    ++it;
  }

  /** Set the deformation field in the transform. */
  this->m_IntermediaryDeformationFieldTransform->SetCoefficientVectorImage(intermediaryDeformationField);

  /** Set to initialized. */
  this->m_Initialized = true;

} // end InitializeDeformationFields()


/**
 * *********************** TransformPoint ***********************
 */

template <class TAnyITKTransform>
auto
DeformationFieldRegulizer<TAnyITKTransform>::TransformPoint(const InputPointType & inputPoint) const -> OutputPointType
{
  /** Get the outputpoint of any ITK Transform and the deformation field. */
  OutputPointType oppAnyT, oppDF, opp;
  oppAnyT = this->Superclass::TransformPoint(inputPoint);
  oppDF = this->m_IntermediaryDeformationFieldTransform->TransformPoint(inputPoint);

  /** Add them: don't forget to subtract ipp. */
  for (unsigned int i = 0; i < OutputSpaceDimension; ++i)
  {
    opp[i] = oppAnyT[i] + oppDF[i] - inputPoint[i];
  }

  /** Return a value. */
  return opp;

} // end TransformPoint()


/**
 * ******** UpdateIntermediaryDeformationFieldTransform *********
 */

template <class TAnyITKTransform>
void
DeformationFieldRegulizer<TAnyITKTransform>::UpdateIntermediaryDeformationFieldTransform(
  typename VectorImageType::Pointer vecImage)
{
  /** Set the vecImage (which is allocated elsewhere) and put it in
   * IntermediaryDeformationFieldTransform (where it is copied and split up).
   */
  this->m_IntermediaryDeformationFieldTransform->SetCoefficientVectorImage(vecImage);

} // end UpdateIntermediaryDeformationFieldTransform()


} // end namespace itk

#endif // end #ifndef itkDeformationFieldRegulizer_hxx
