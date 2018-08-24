/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkPointToSurfaceDistanceMetric_hxx
#define __itkPointToSurfaceDistanceMetric_hxx

#include "itkPointToSurfaceDistanceMetric.h"

namespace itk
{

/**
 * ******************* Constructor *******************
 */

template< class TFixedPointSet, class TMovingPointSet >
PointToSurfaceDistanceMetric< TFixedPointSet, TMovingPointSet >
::PointToSurfaceDistanceMetric()
{

} // end Constructor

template< class TFixedPointSet, class TMovingPointSet >
void PointToSurfaceDistanceMetric< TFixedPointSet, TMovingPointSet >
::Initialize( void ) throw ( ExceptionObject )
{
    /** Initialize transform, interpolator, etc.
    Superclass::Initialize();

    /***********************ITK Stuffs******************************/
    /** Sanity checks. */

    if ((m_SegFileIn.length()<=0)&&(m_DTFileIn.length()<=0))
    {
        itkExceptionMacro( << "Neither Distance Transform nor Segmentation file is not found" );
    }

    if ((m_SegFileIn.length()>0)&&(m_DTFileIn.length()>0))
    {
        itkExceptionMacro( << "Distance Transform and Segmentation files can not be loaded at the same time" );
    }
    /***********************ITK Stuffs******************************/

    if ((m_SegFileIn.length()>0))
    {
        /***********************Absolute DT computation*****************/
        m_DTfilter = DtFilterType::New();
        m_DTfilter2 = DtFilterType::New();
        m_NegateFilter=NegateFilterType::New();
        m_scaler = RescalerType::New();
        m_AddImage=AddImageFilt::New();
        ADTimage=ImageType::New();
        m_segReader = SegReaderType::New();
        m_segReader2 = SegReaderType::New();
        m_segReader->SetFileName( m_SegFileIn );
        m_segReader2->SetFileName( m_SegFileIn );

        /******This values for rescaling were put trivially***********/
        m_scaler->SetOutputMaximum( 10L );//These values should be changed after comprehensive experiments
        m_scaler->SetOutputMinimum(     0L );
        /******This values for rescaling were put trivially***********/

        m_segReader->Update();
        m_segReader2->Update();

        m_DTfilter->SetInput( m_segReader->GetOutput() );
        m_NegateFilter->SetInput( m_segReader2->GetOutput() );
        m_DTfilter2->SetInput( m_NegateFilter->GetOutput() );

        m_AddImage->SetInput1(m_DTfilter->GetOutput());
        m_AddImage->SetInput2(m_DTfilter2->GetOutput());
        m_AddImage->Update();

        m_scaler->SetInput( m_AddImage->GetOutput() );
        m_scaler->Update();

        ADTimage=m_scaler->GetOutput();
        /***********************Absolute DT computation*****************/
    }

    if ((m_DTFileIn.length()>0))
    {
        ADTimage=ImageType::New();
        m_dtReader = DTReaderType::New();
        m_dtReader->SetFileName( m_DTFileIn );
        m_dtReader->Update();
        ADTimage=m_dtReader->GetOutput();
    }

    interpolator = InterpolatorType::New();
    interpolator->SetSplineOrder(3);
    interpolator->SetInputImage(ADTimage);

    if (m_DTFileOut.length()>0)
	{
            m_writer = WriterType::New();
            m_writer->SetFileName( m_DTFileOut);
            m_writer->SetInput( ADTimage );
            m_writer->Update();
	}

	// Write an ADT image outside, just for checking
    /*****************Absolute DT computation*****************/

} // end Initialize()


/**
 * ******************* GetValue *******************
 */

template< class TFixedPointSet, class TMovingPointSet >
typename PointToSurfaceDistanceMetric< TFixedPointSet, TMovingPointSet >::MeasureType
PointToSurfaceDistanceMetric< TFixedPointSet, TMovingPointSet >
::GetValue( const TransformParametersType & parameters ) const
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
    MeasureType     measure = NumericTraits< MeasureType >::Zero;

    OutputPointType fixedPoint, mappedPoint;

    /** Make sure the transform parameters are up to date. */
    this->SetTransformParameters( parameters );

    /** Create iterators. */
    PointIterator pointItFixed  = fixedPointSet->GetPoints()->Begin();
    PointIterator pointEnd      = fixedPointSet->GetPoints()->End();
    /***************Initialization of some variables****************/


    /*********************ADT cost computation**********************/
    /** Loop over the corresponding points. */
    while( pointItFixed != pointEnd )
    {
        /** Get the current corresponding points. */
        fixedPoint  = pointItFixed.Value();

        /** Transform point and check if it is inside the B-spline support region. */
        mappedPoint = this->m_Transform->TransformPoint( fixedPoint );

        bool sampleOk = true;
        if( sampleOk )
        {
            double meas=interpolator->Evaluate(mappedPoint);
            this->m_NumberOfPointsCounted++;
            measure += meas*meas;
        } // end if sampleOk

        ++pointItFixed;
    } // end loop over all corresponding points
    /*********************ADT cost computation**********************/

    if( (this->m_NumberOfPointsCounted > 0) && (this->m_AvPointWeigh)) measure/=this->m_NumberOfPointsCounted;

    return (measure);

} // end GetValue()


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

} // end GetDerivative()

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
    MeasureType     measure = NumericTraits< MeasureType >::Zero;

    OutputPointType fixedPoint, mappedPoint;

    /** Make sure the transform parameters are up to date. */
    this->SetTransformParameters( parameters );

    this->BeforeThreadedGetValueAndDerivative( parameters );

    /** Create iterators. */
    PointIterator pointItFixed  = fixedPointSet->GetPoints()->Begin();
    PointIterator pointEnd      = fixedPointSet->GetPoints()->End();

    derivative = DerivativeType( this->GetNumberOfParameters() );

    derivative.Fill( NumericTraits< DerivativeValueType >::Zero );
    /***************Initialization of some variables****************/
    TransformJacobianType jacobian;

    typename InterpolatorType::CovariantVectorType covVector;
    /** Loop over the corresponding points. */
    while( pointItFixed != pointEnd )
    {
		
        NonZeroJacobianIndicesType nzji( this->m_Transform->GetNumberOfNonZeroJacobianIndices() );

        /** Get the current corresponding points. */
        fixedPoint  = pointItFixed.Value();

        /** Transform point and check if it is inside the B-spline support region. */
        mappedPoint = this->m_Transform->TransformPoint( fixedPoint );

        bool sampleOk = true;
        if( sampleOk )
            {
                double meas=interpolator->Evaluate(mappedPoint);
                double Coeff=2*meas;
                this->m_NumberOfPointsCounted++;
                measure += meas*meas;//*interpolator->Evaluate(mappedPoint);

                /** Get the TransformJacobian dT/dmu. */
                this->m_Transform->GetJacobian( fixedPoint, jacobian, nzji );

                VnlVectorType diff=fixedPoint.GetVnlVector();

                /********Finding the image derivative***********/
                covVector=interpolator->EvaluateDerivative(mappedPoint);
                /********Finding the image derivative***********/

                /***********Converting to derivative Type*******/
                for(int i=0;i<covVector.GetCovariantVectorDimension();i++)
                    {
                        diff.put(0,covVector.GetElement(0));
                        diff.put(1,covVector.GetElement(1));
                        diff.put(2,covVector.GetElement(2));
                    }

                /***********Converting to derivative Type*******/
                if( nzji.size() == this->GetNumberOfParameters() )
                    {
                        /** Loop over all Jacobians. */
                        derivative +=Coeff*diff * jacobian;
                    }
                else
                    {
                        /** Only pick the nonzero Jacobians. */
                        for( unsigned int i = 0; i < nzji.size(); ++i )
                        {
                            const unsigned int index  = nzji[ i ];
                            VnlVectorType      column = jacobian.get_column( i );
                            derivative[ index ] += Coeff*dot_product( diff, column );
                        }
                    }

                } // end if sampleOk

           ++pointItFixed;

    } // end loop over all corresponding points

     value       = measure;
    /** Taking average of grad and measure*/
    if( (this->m_NumberOfPointsCounted > 0) && (this->m_AvPointWeigh))
    {
        derivative /= this->m_NumberOfPointsCounted;
        value      /= this->m_NumberOfPointsCounted;
    }

} // end GetValueAndDerivative()

	/** Set input Segmented Image File **/
template< class TFixedPointSet, class TMovingPointSet >
void
PointToSurfaceDistanceMetric< TFixedPointSet, TMovingPointSet >
::SetSegImageIn( const std::string  str ) 
{
	this->m_SegFileIn=str;
}

/** Set input Distance Transform Image File **/
template< class TFixedPointSet, class TMovingPointSet >
void
PointToSurfaceDistanceMetric< TFixedPointSet, TMovingPointSet >
::SetDTImageIn( const std::string  str )
{
this->m_DTFileIn=str;
}
	/** Set output Distance Transform File **/
template< class TFixedPointSet, class TMovingPointSet >
void
PointToSurfaceDistanceMetric< TFixedPointSet, TMovingPointSet >
::SetDTImageOut( const std::string  str ) 
{
	this->m_DTFileOut=str;
}

/** Get Distance Transform Image*/
template< class TFixedPointSet, class TMovingPointSet >
typename PointToSurfaceDistanceMetric< TFixedPointSet, TMovingPointSet >::ImageType::ConstPointer
PointToSurfaceDistanceMetric< TFixedPointSet, TMovingPointSet >
::GetDTImage(void)
{
    return static_cast<typename PointToSurfaceDistanceMetric< TFixedPointSet, TMovingPointSet >::ImageType::ConstPointer> (ADTimage);
}

} // end namespace itk

#endif // end #ifndef __itkPointToSurfaceDistanceMetric_hxx
