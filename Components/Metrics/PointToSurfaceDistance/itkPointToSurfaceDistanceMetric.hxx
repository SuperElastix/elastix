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
#ifndef __itkPointToSurfaceDistanceMetric_hxx
#define __itkPointToSurfaceDistanceMetric_hxx

#include "itkPointToSurfaceDistanceMetric.h"
#include <iterator>

namespace itk
{

/**
 * ******************* Constructor *******************
 */

template< class TFixedPointSet, class TMovingPointSet >
PointToSurfaceDistanceMetric< TFixedPointSet, TMovingPointSet >
::PointToSurfaceDistanceMetric()
  : m_internalDistanceTransformImage{ nullptr }
  , m_interpolator{ nullptr }

{
}

template< class TFixedPointSet, class TMovingPointSet >
void PointToSurfaceDistanceMetric< TFixedPointSet, TMovingPointSet >
::Initialize()
{
  /** Initialize transform, interpolator, etc.
  Superclass::Initialize();
  
  /***********************ITK Stuffs******************************/
  /** Sanity checks. */
  
  if ((m_segmentationFileIn.length() <= 0) && (m_distanceTransformFileIn.length() <= 0))
  {
      itkExceptionMacro( << "Neither Distance Transform nor Segmentation file is not found" );
  }
  
  if ((m_segmentationFileIn.length() > 0) && (m_distanceTransformFileIn.length() > 0))
  {
      itkExceptionMacro( << "Distance Transform and Segmentation files can not be loaded at the same time" );
  }
    /***********************ITK Stuffs******************************/
  try
  {
    if ((m_segmentationFileIn.length() > 0))
    {
      /***********************Absolute DT computation*****************/
      m_internalDistanceTransformImage = ImageType::New();
      auto DTfilter = DtFilterType::New();
      auto DTfilter2 = DtFilterType::New();
      auto NegateFilter = NegateFilterType::New();
      auto scaler = RescalerType::New();
      auto AddImage = AddImageFilt::New();
      auto segReader = SegReaderType::New();
      auto segReader2 = SegReaderType::New();
      segReader->SetFileName( m_segmentationFileIn );
      segReader2->SetFileName( m_segmentationFileIn );
      
      /******This values for rescaling were put trivially***********/
      scaler->SetOutputMaximum( 10L );//These values should be changed after comprehensive experiments
      scaler->SetOutputMinimum( 0L );
      /******This values for rescaling were put trivially***********/
      
      segReader->Update();
      segReader2->Update();
      
      DTfilter->SetInput( segReader->GetOutput() );
      NegateFilter->SetInput( segReader2->GetOutput() );
      DTfilter2->SetInput( NegateFilter->GetOutput() );
      
      AddImage->SetInput1(DTfilter->GetOutput());
      AddImage->SetInput2(DTfilter2->GetOutput());
      AddImage->Update();
      
      scaler->SetInput( AddImage->GetOutput() );
      scaler->Update();
      
      m_internalDistanceTransformImage = scaler->GetOutput();
      m_internalDistanceTransformImage->DisconnectPipeline();
      /***********************Absolute DT computation*****************/
    }
    
    // Read an ADT image outside.
    if ((m_distanceTransformFileIn.length()>0))
    {
      m_internalDistanceTransformImage = ImageType::New();
      auto dtReader = DTReaderType::New();
      dtReader->SetFileName( m_distanceTransformFileIn );
      dtReader->Update();
      m_internalDistanceTransformImage = dtReader->GetOutput();
    }
    
    // Write an ADT image outside, just for checking.
    if (m_distanceTransformFileOut.length()>0)
    {
      auto writer = WriterType::New();
      writer->SetFileName( m_distanceTransformFileOut);
      writer->SetInput( m_internalDistanceTransformImage );
      writer->Update();
    }
  }
	catch(...)
	{
    itkExceptionMacro(<< "There is a problem in the creation of distance transform image.");
	}

  if (m_internalDistanceTransformImage)
  {
    m_interpolator = InterpolatorType::New();
    m_interpolator->SetSplineOrder(3);
    m_interpolator->SetInputImage(m_internalDistanceTransformImage);
  }
  else
  {
    itkExceptionMacro(<< "There is no distance transform image.");
  }
}

/**
 * ******************* GetValue *******************
 */

template< class TFixedPointSet, class TMovingPointSet >
auto
PointToSurfaceDistanceMetric< TFixedPointSet, TMovingPointSet >
::GetValue( const TransformParametersType & parameters ) const -> MeasureType
{
	/***********************ITK Stuffs******************************/
    /** Sanity checks. */
    auto fixedPointSet = this->GetFixedPointSet();
    if( !fixedPointSet )
    {
        itkExceptionMacro( << "Fixed point set has not been assigned" );
    }

    /***********************ITK Stuffs******************************/
    auto measure = NumericTraits< MeasureType >::Zero;

    /** Make sure the transform parameters are up to date. */
    this->SetTransformParameters( parameters );
    
    /*********************ADT cost computation**********************/
    /** Loop over the corresponding points. */
    {
      this->m_NumberOfPointsCounted = 0;
      OutputPointType fixedPoint, mappedPoint;
      const auto pointItFixedEnd = fixedPointSet->GetPoints()->End();
      for (auto pointItFixed = fixedPointSet->GetPoints()->Begin(); pointItFixed != pointItFixedEnd; ++pointItFixed)
      {
        /** Get the current corresponding points. */
        fixedPoint = pointItFixed.Value();

        /** Transform point and check if it is inside the B-spline support region. */
        mappedPoint = this->m_Transform->TransformPoint(fixedPoint);

        double meas = m_interpolator->Evaluate(mappedPoint);
        ++this->m_NumberOfPointsCounted;
        measure += meas * meas;
      } // end loop over all corresponding points
    }
    /*********************ADT cost computation**********************/

    if ((this->m_NumberOfPointsCounted > 0) && (this->m_AvPointWeigh))
    {
      measure /= this->m_NumberOfPointsCounted;
    }

    return (measure);
}

/**
 * ******************* GetDerivative *******************
 */

template< class TFixedPointSet, class TMovingPointSet >
void
PointToSurfaceDistanceMetric< TFixedPointSet, TMovingPointSet >
::GetDerivative( const TransformParametersType & parameters, DerivativeType & derivative ) const
{
    MeasureType dummyvalue = NumericTraits< MeasureType >::Zero;
    this->GetValueAndDerivative( parameters, dummyvalue, derivative );
}

/**
 * ******************* GetValueAndDerivative *******************
 */

template< class TFixedPointSet, class TMovingPointSet >
void
PointToSurfaceDistanceMetric< TFixedPointSet, TMovingPointSet >
::GetValueAndDerivative( const TransformParametersType & parameters, MeasureType & value, DerivativeType & derivative ) const
{
    /***********************ITK Stuffs******************************/
    /** Sanity checks. */

    FixedPointSetConstPointer fixedPointSet = this->GetFixedPointSet();
    if( !fixedPointSet )
    {
        itkExceptionMacro( << "Fixed point set has not been assigned" );
    }

    /***********************ITK Stuffs******************************/

    /***************Initialization of some variables****************/
    /** Initialize some variables. */
    this->m_NumberOfPointsCounted = 0;
    MeasureType measure = NumericTraits< MeasureType >::Zero;

    /** Make sure the transform parameters are up to date. */
    this->SetTransformParameters( parameters );

    this->BeforeThreadedGetValueAndDerivative( parameters );

    derivative = DerivativeType( this->GetNumberOfParameters() );
    derivative.Fill( NumericTraits< DerivativeValueType >::Zero );
    /***************Initialization of some variables****************/
    TransformJacobianType jacobian;
    typename InterpolatorType::CovariantVectorType covVector;

    {
      OutputPointType fixedPoint, mappedPoint;
      /** Loop over the corresponding points. */
      const auto pointEnd = fixedPointSet->GetPoints()->End();
      for (auto pointItFixed = fixedPointSet->GetPoints()->Begin(); pointItFixed != pointEnd; ++pointItFixed)
      {

        NonZeroJacobianIndicesType nzji(this->m_Transform->GetNumberOfNonZeroJacobianIndices());
        /** Get the current corresponding points. */
        fixedPoint = pointItFixed.Value();
        /** Transform point and check if it is inside the B-spline support region. */
        mappedPoint = this->m_Transform->TransformPoint(fixedPoint);

        const auto meas = m_interpolator->Evaluate(mappedPoint);
        const auto Coeff = 2 * meas;
        measure += meas * meas;//*interpolator->Evaluate(mappedPoint);

        /** Get the TransformJacobian dT/dmu. */
        this->m_Transform->GetJacobian(fixedPoint, jacobian, nzji);

        VnlVectorType diff = fixedPoint.GetVnlVector();

        /********Finding the image derivative***********/
        covVector = m_interpolator->EvaluateDerivative(mappedPoint);
        /********Finding the image derivative***********/

        /***********Converting to derivative Type*******/
        for (auto i = 0u; i < covVector.GetCovariantVectorDimension(); i++)
        {
          diff.put(0, covVector.GetElement(0));
          diff.put(1, covVector.GetElement(1));
          diff.put(2, covVector.GetElement(2));
        }

        const auto sizeJI = nzji.size();
        /***********Converting to derivative Type*******/
        if (sizeJI == this->GetNumberOfParameters())
        {
          /** Loop over all Jacobians. */
          derivative += Coeff * diff * jacobian;
        }
        else
        {
          /** Only pick the nonzero Jacobians. */
          for (auto i = 0u; i < sizeJI; ++i)
          {
            const unsigned int index = nzji[i];
            VnlVectorType column = jacobian.get_column(i);
            derivative[index] += Coeff * dot_product(diff, column);
          }
        }

        ++this->m_NumberOfPointsCounted;
      } // end loop over all corresponding points
    }

    value = measure;
    /** Taking average of grad and measure*/
    if( (this->m_NumberOfPointsCounted > 0) && (this->m_AvPointWeigh))
    {
        derivative /= this->m_NumberOfPointsCounted;
        value /= this->m_NumberOfPointsCounted;
    }

}

/** Set input Segmented Image File **/
template< class TFixedPointSet, class TMovingPointSet >
void
PointToSurfaceDistanceMetric< TFixedPointSet, TMovingPointSet >
::SetSegImageIn( const std::string  str ) 
{
	this->m_segmentationFileIn = str;
}

/** Set input Distance Transform Image File **/
template< class TFixedPointSet, class TMovingPointSet >
void
PointToSurfaceDistanceMetric< TFixedPointSet, TMovingPointSet >
::SetDTImageIn( const std::string  str )
{
  this->m_distanceTransformFileIn = str;
}

/** Set output Distance Transform File **/
template< class TFixedPointSet, class TMovingPointSet >
void
PointToSurfaceDistanceMetric< TFixedPointSet, TMovingPointSet >
::SetDTImageOut( const std::string  str ) 
{
	this->m_distanceTransformFileOut = str;
}

/** Get Distance Transform Image*/
template< class TFixedPointSet, class TMovingPointSet >
auto
PointToSurfaceDistanceMetric< TFixedPointSet, TMovingPointSet >
::GetDTImage() const -> ConstDTimage
{
    return static_cast<typename PointToSurfaceDistanceMetric< TFixedPointSet, TMovingPointSet >::ImageType::ConstPointer>(m_internalDistanceTransformImage);
}

} // end namespace itk

#endif // end #ifndef __itkPointToSurfaceDistanceMetric_hxx
