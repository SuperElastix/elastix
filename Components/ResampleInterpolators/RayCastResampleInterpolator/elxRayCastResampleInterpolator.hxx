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

#ifndef elxRayCastResampleInterpolator_hxx
#define elxRayCastResampleInterpolator_hxx

#include "elxRayCastResampleInterpolator.h"
#include "itkImageFileWriter.h"

namespace elastix
{

/*
 * ***************** BeforeAll *****************
 */

template <class TElastix>
void
RayCastResampleInterpolator<TElastix>::InitializeRayCastInterpolator()
{

  this->m_CombinationTransform = CombinationTransformType::New();
  this->m_CombinationTransform->SetUseComposition(true);

  this->m_PreTransform = EulerTransformType::New();
  unsigned int            numberofparameters = this->m_PreTransform->GetNumberOfParameters();
  TransformParametersType preParameters(numberofparameters);
  preParameters.Fill(0.0);

  for (unsigned int i = 0; i < numberofparameters; ++i)
  {
    bool ret =
      this->GetConfiguration()->ReadParameter(preParameters[i], "PreParameters", this->GetComponentLabel(), i, 0);
    if (!ret)
    {
      std::cerr << " Error, not enough PreParameters are given" << std::endl;
    }
  }

  typename EulerTransformType::InputPointType centerofrotation;
  centerofrotation.Fill(0.0);

  for (unsigned int i = 0; i < this->m_Elastix->GetMovingImage()->GetImageDimension(); ++i)
  {
    this->GetConfiguration()->ReadParameter(
      centerofrotation[i], "CenterOfRotationPoint", this->GetComponentLabel(), i, 0);
  }

  this->m_PreTransform->SetParameters(preParameters);
  this->m_PreTransform->SetCenter(centerofrotation);
  this->m_CombinationTransform->SetInitialTransform(this->m_PreTransform);
  this->m_CombinationTransform->SetCurrentTransform(this->m_Elastix->GetElxTransformBase()->GetAsITKBaseType());
  this->SetTransform(this->m_CombinationTransform);
  this->SetInputImage(this->m_Elastix->GetMovingImage());

  PointType focalPoint;
  focalPoint.Fill(0.0);

  for (unsigned int i = 0; i < this->m_Elastix->GetFixedImage()->GetImageDimension(); ++i)
  {
    bool ret = this->GetConfiguration()->ReadParameter(focalPoint[i], "FocalPoint", this->GetComponentLabel(), i, 0);
    if (!ret)
    {
      std::cerr << " Error, FocalPoint not assigned" << std::endl;
    }
  }

  this->SetFocalPoint(focalPoint);
  this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->SetTransform(this->m_CombinationTransform);
  double threshold = 0.;
  this->GetConfiguration()->ReadParameter(threshold, "Threshold", 0);
  this->SetThreshold(threshold);

} // end InitializeRayCastInterpolator()


/*
 * ***************** BeforeAll *****************
 */

template <class TElastix>
int
RayCastResampleInterpolator<TElastix>::BeforeAll()
{

  // Check if 2D-3D
  if (this->m_Elastix->GetFixedImage()->GetImageDimension() != 3)
  {
    itkExceptionMacro(<< "The RayCastInterpolator expects the fixed image to be 3D.");
    return 1;
  }
  if (this->m_Elastix->GetMovingImage()->GetImageDimension() != 3)
  {
    itkExceptionMacro(<< "The RayCastInterpolator expects the moving image to be 3D.");
    return 1;
  }

  return 0;

} // end BeforeAll()


/*
 * ***************** BeforeRegistration *****************
 */

template <class TElastix>
void
RayCastResampleInterpolator<TElastix>::BeforeRegistration()
{

  this->InitializeRayCastInterpolator();

} // end BeforeRegistration()


/*
 * ***************** ReadFromFile *****************
 */

template <class TElastix>
void
RayCastResampleInterpolator<TElastix>::ReadFromFile()
{

  /** Call ReadFromFile of the ResamplerBase. */
  this->Superclass2::ReadFromFile();
  this->InitializeRayCastInterpolator();

} // end ReadFromFile()


/**
 * ******************* CreateDerivedTransformParametersMap ******************************
 */

template <class TElastix>
auto
RayCastResampleInterpolator<TElastix>::CreateDerivedTransformParametersMap() const -> ParameterMapType
{
  return { { "FocalPoint", Conversion::ToVectorOfStrings(this->GetFocalPoint()) },
           { "PreParameters", Conversion::ToVectorOfStrings(this->m_PreTransform->GetParameters()) },
           { "Threshold", { Conversion::ToString(this->GetThreshold()) } } };

} // end CreateDerivedTransformParametersMap()

} // end namespace elastix

#endif // end #ifndef elxRayCastResampleInterpolator_hxx
