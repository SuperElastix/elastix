#ifndef _itkCombinedImageToImageMetric_txx
#define _itkCombinedImageToImageMetric_txx

#include "itkCombinedImageToImageMetric.h"


namespace itk
{

  /**
	 * ********************* Constructor ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    CombinedImageToImageMetric<TFixedImage,TMovingImage>
    ::CombinedImageToImageMetric()
  {
    this->m_NumberOfMetrics = 0;
  } // end Constructor


  /**
	 * ********************* PrintSelf ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    CombinedImageToImageMetric<TFixedImage,TMovingImage>
    ::PrintSelf( std::ostream& os, Indent indent ) const
  {
		/** Call the superclass' PrintSelf. */
		Superclass::PrintSelf( os, indent );
		
		/** Add debugging information. */
		os << indent << "NumberOfMetrics: "
			<< this->m_NumberOfMetrics << std::endl;
    os << indent << "Metric pointer, weight, value:" << std::endl;
    for ( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
    {
      os << indent << "Metric " << i << ": "
        << this->m_Metrics[ i ].GetPointer() << ", "
        << this->m_MetricWeights[ i ] << ", "
        << this->m_MetricValues[ i ] << std::endl;
    }
				
	} // end PrintSelf


  /**
	 * ********************* SetNumberOfMetrics ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    CombinedImageToImageMetric<TFixedImage,TMovingImage>
    ::SetNumberOfMetrics( unsigned int count )
  {
    if ( count != this->m_Metrics.size() )
    {
      this->m_NumberOfMetrics = count;
      this->m_Metrics.resize( count );
      this->m_MetricWeights.resize( count );
      this->m_MetricValues.resize( count );
      this->Modified();
    }

  } // end SetNumberOfMetrics


  /**
	 * ********************* SetMetric ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    CombinedImageToImageMetric<TFixedImage,TMovingImage>
    ::SetMetric( SingleValuedCostFunctionType * metric, unsigned int pos )
  {
    if ( pos > this->m_NumberOfMetrics )
    {
      this->SetNumberOfMetrics( pos );
      this->Modified();
    }

    if ( metric != this->m_Metrics[ pos ] )
    {
      this->m_Metrics[ pos ] = metric;
      this->Modified();
    }
    
  } // end SetMetric
  

  /**
	 * ********************* GetMetric ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    typename CombinedImageToImageMetric<TFixedImage,TMovingImage>::SingleValuedCostFunctionType * 
    CombinedImageToImageMetric<TFixedImage,TMovingImage>
    ::GetMetric( unsigned int pos )
  {
    if ( pos > this->m_NumberOfMetrics )
    {
      return 0;
    }
    else
    {
      return this->m_Metrics[ pos ];
    }
    
  } // end GetMetric


  /**
	 * ********************* SetMetricWeight ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    CombinedImageToImageMetric<TFixedImage,TMovingImage>
    ::SetMetricWeight( double weight, unsigned int pos )
  {
    if ( pos > this->m_NumberOfMetrics )
    {
      this->SetNumberOfMetrics( pos );
      this->Modified();
    }

    if ( weight != this->m_MetricWeights[ pos ] )
    {
      this->m_MetricWeights[ pos ] = weight;
      this->Modified();
    }
    
  } // end SetMetricsWeight


  /**
	 * ********************* GetMetricWeight ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    double
    CombinedImageToImageMetric<TFixedImage,TMovingImage>
    ::GetMetricWeight( unsigned int pos )
  {
    if ( pos > this->m_NumberOfMetrics )
    {
      return 0.0;
    }
    else
    {
      return this->m_MetricWeights[ pos ];
    }
    
  } // end GetMetricWeight


  /**
	 * ********************* GetMetricValue ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    typename CombinedImageToImageMetric<TFixedImage,TMovingImage>::MeasureType
    CombinedImageToImageMetric<TFixedImage,TMovingImage>
    ::GetMetricValue( unsigned int pos )
  {
    if ( pos > this->m_NumberOfMetrics )
    {
      return 0.0;
    }
    else
    {
      return this->m_MetricValues[ pos ];
    }
    
  } // end GetMetricValue


  /**
	 * ********************* SetTransform ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    CombinedImageToImageMetric<TFixedImage,TMovingImage>
    ::SetTransform( TransformType *_arg )
  {
    this->Superclass::SetTransform( _arg );

    for ( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
    {
      MetricType * testPtr = dynamic_cast<MetricType *>( this->m_Metrics[ i ].GetPointer() );
      if ( testPtr )
      {
        testPtr->SetTransform( _arg );
      }
    }
  } // end SetTransform


  /**
	 * ********************* SetInterpolator ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    CombinedImageToImageMetric<TFixedImage,TMovingImage>
    ::SetInterpolator( InterpolatorType *_arg )
  {
    this->Superclass::SetInterpolator( _arg );

    for ( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
    {
      MetricType * testPtr = dynamic_cast<MetricType *>( this->m_Metrics[ i ].GetPointer() );
      if ( testPtr )
      {
        testPtr->SetInterpolator( _arg );
      }
    }
  } // end SetInterpolator

  
  /**
	 * ********************* SetFixedImage ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    CombinedImageToImageMetric<TFixedImage,TMovingImage>
    ::SetFixedImage( const FixedImageType *_arg )
  {
    this->Superclass::SetFixedImage( _arg );

    for ( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
    {
      MetricType * testPtr = dynamic_cast<MetricType *>( this->m_Metrics[ i ].GetPointer() );
      if ( testPtr )
      {
        testPtr->SetFixedImage( _arg );
      }
    }
  } // end SetFixedImage


  /**
	 * ********************* SetFixedImageMask ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    CombinedImageToImageMetric<TFixedImage,TMovingImage>
    ::SetFixedImageMask( FixedImageMaskType *_arg )
  {
    this->Superclass::SetFixedImageMask( _arg );

    for ( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
    {
      MetricType * testPtr = dynamic_cast<MetricType *>( this->m_Metrics[ i ].GetPointer() );
      if ( testPtr )
      {
        testPtr->SetFixedImageMask( _arg );
      }
    }
  } // end SetFixedImageMask


  /**
	 * ********************* SetFixedImageRegion ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    CombinedImageToImageMetric<TFixedImage,TMovingImage>
    ::SetFixedImageRegion( const FixedImageRegionType _arg )
  {
    this->Superclass::SetFixedImageRegion( _arg );

    for ( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
    {
      MetricType * testPtr = dynamic_cast<MetricType *>( this->m_Metrics[ i ].GetPointer() );
      if ( testPtr )
      {
        testPtr->SetFixedImageRegion( _arg );
      }
    }
  } // end SetFixedImageRegion


  
  /**
	 * ********************* SetMovingImage ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    CombinedImageToImageMetric<TFixedImage,TMovingImage>
    ::SetMovingImage( const MovingImageType *_arg )
  {
    this->Superclass::SetMovingImage( _arg );

    for ( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
    {
      MetricType * testPtr = dynamic_cast<MetricType *>( this->m_Metrics[ i ].GetPointer() );
      if ( testPtr )
      {
        testPtr->SetMovingImage( _arg );
      }
    }
  } // end SetMovingImage

  
  /**
	 * ********************* SetMovingImageMask ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    CombinedImageToImageMetric<TFixedImage,TMovingImage>
    ::SetMovingImageMask( MovingImageMaskType *_arg )
  {
    this->Superclass::SetMovingImageMask( _arg );

    for ( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
    {
      MetricType * testPtr = dynamic_cast<MetricType *>( this->m_Metrics[ i ].GetPointer() );
      if ( testPtr )
      {
        testPtr->SetMovingImageMask( _arg );
      }
    }
  } // end SetMovingImageMask


  /**
	 * ********************* SetComputeGradient ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    CombinedImageToImageMetric<TFixedImage,TMovingImage>
    ::SetComputeGradient( const bool _arg )
  {
    this->Superclass::SetComputeGradient( _arg );

    for ( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
    {
      MetricType * testPtr = dynamic_cast<MetricType *>( this->m_Metrics[ i ].GetPointer() );
      if ( testPtr )
      {
        testPtr->SetComputeGradient( _arg );
      }
    }
  } // end SetComputeGradient

  
  /**
	 * ********************* Initialize ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    CombinedImageToImageMetric<TFixedImage,TMovingImage>
    ::Initialize( void )
  {
    this->Superclass::Initialize();

    for ( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
    {
      MetricType * testPtr = dynamic_cast<MetricType *>( this->m_Metrics[ i ].GetPointer() );
      if ( testPtr )
      {
        testPtr->Initialize();
      }
    }
  } // end Initialize


  /**
	 * ********************* SetImageSampler ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    CombinedImageToImageMetric<TFixedImage,TMovingImage>
    ::SetImageSampler( ImageSamplerType * _arg )
  {
    this->Superclass::SetImageSampler( _arg );

    for ( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
    {
      AdvancedMetricType * testPtr =
        dynamic_cast<AdvancedMetricType *>( this->m_Metrics[ i ].GetPointer() );
      if ( testPtr )
      {
        testPtr->SetImageSampler( _arg );
      }
    }
  } // end SetImageSampler

  
  /**
	 * ********************* SetRequiredRatioOfValidSamples ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    CombinedImageToImageMetric<TFixedImage,TMovingImage>
    ::SetRequiredRatioOfValidSamples( const double _arg )
  {
    this->Superclass::SetRequiredRatioOfValidSamples( _arg );

    for ( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
    {
      AdvancedMetricType * testPtr =
        dynamic_cast<AdvancedMetricType *>( this->m_Metrics[ i ].GetPointer() );
      if ( testPtr )
      {
        testPtr->SetRequiredRatioOfValidSamples( _arg );
      }
    }
  } // end SetRequiredRatioOfValidSamples


  /**
	 * ********************* SetUseDifferentiableOverlap ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    CombinedImageToImageMetric<TFixedImage,TMovingImage>
    ::SetUseDifferentiableOverlap( const bool _arg )
  {
    this->Superclass::SetUseDifferentiableOverlap( _arg );

    for ( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
    {
      AdvancedMetricType * testPtr =
        dynamic_cast<AdvancedMetricType *>( this->m_Metrics[ i ].GetPointer() );
      if ( testPtr )
      {
        testPtr->SetUseDifferentiableOverlap( _arg );
      }
    }
  } // end SetUseDifferentiableOverlap


  /**
	 * ********************* SetMovingImageMaskInterpolationOrder ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    CombinedImageToImageMetric<TFixedImage,TMovingImage>
    ::SetMovingImageMaskInterpolationOrder( unsigned int _arg )
  {
    this->Superclass::SetMovingImageMaskInterpolationOrder( _arg );

    for ( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
    {
      AdvancedMetricType * testPtr =
        dynamic_cast<AdvancedMetricType *>( this->m_Metrics[ i ].GetPointer() );
      if ( testPtr )
      {
        testPtr->SetMovingImageMaskInterpolationOrder( _arg );
      }
    }
  } // end SetMovingImageMaskInterpolationOrder


  /**
	 * ********************* SetFixedImageLimiter ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    CombinedImageToImageMetric<TFixedImage,TMovingImage>
    ::SetFixedImageLimiter( FixedImageLimiterType * _arg )
  {
    this->Superclass::SetFixedImageLimiter( _arg );

    for ( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
    {
      AdvancedMetricType * testPtr =
        dynamic_cast<AdvancedMetricType *>( this->m_Metrics[ i ].GetPointer() );
      if ( testPtr )
      {
        testPtr->SetFixedImageLimiter( _arg );
      }
    }
  } // end SetFixedImageLimiter


  /**
	 * ********************* SetMovingImageLimiter ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    CombinedImageToImageMetric<TFixedImage,TMovingImage>
    ::SetMovingImageLimiter( MovingImageLimiterType * _arg )
  {
    this->Superclass::SetMovingImageLimiter( _arg );

    for ( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
    {
      AdvancedMetricType * testPtr =
        dynamic_cast<AdvancedMetricType *>( this->m_Metrics[ i ].GetPointer() );
      if ( testPtr )
      {
        testPtr->SetMovingImageLimiter( _arg );
      }
    }
  } // end SetMovingImageLimiter


  /**
	 * ********************* SetFixedLimitRangeRatio ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    CombinedImageToImageMetric<TFixedImage,TMovingImage>
    ::SetFixedLimitRangeRatio( const double _arg )
  {
    this->Superclass::SetFixedLimitRangeRatio( _arg );

    for ( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
    {
      AdvancedMetricType * testPtr =
        dynamic_cast<AdvancedMetricType *>( this->m_Metrics[ i ].GetPointer() );
      if ( testPtr )
      {
        testPtr->SetFixedLimitRangeRatio( _arg );
      }
    }
  } // end SetFixedLimitRangeRatio

  
  /**
	 * ********************* SetMovingLimitRangeRatio ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    CombinedImageToImageMetric<TFixedImage,TMovingImage>
    ::SetMovingLimitRangeRatio( const double _arg )
  {
    this->Superclass::SetMovingLimitRangeRatio( _arg );

    for ( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
    {
      AdvancedMetricType * testPtr =
        dynamic_cast<AdvancedMetricType *>( this->m_Metrics[ i ].GetPointer() );
      if ( testPtr )
      {
        testPtr->SetMovingLimitRangeRatio( _arg );
      }
    }
  } // end SetMovingLimitRangeRatio


  /**
	 * ********************* GetValue ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    typename CombinedImageToImageMetric<TFixedImage,TMovingImage>::MeasureType
    CombinedImageToImageMetric<TFixedImage,TMovingImage>
    ::GetValue( const ParametersType & parameters ) const
  {
    /** Add all metric values. */
    MeasureType measure = NumericTraits< MeasureType >::Zero;
    for ( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
    {
      this->m_MetricValues[ i ] =
        this->m_MetricWeights[ i ] * this->m_Metrics[ i ]->GetValue( parameters );
      measure += this->m_MetricValues[ i ];
    }

    /** Return a value. */
    return measure;

  } // end GetValue


  /**
	 * ********************* GetDerivative ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    CombinedImageToImageMetric<TFixedImage,TMovingImage>
    ::GetDerivative( const ParametersType & parameters,
      DerivativeType & derivative ) const
  {
    /** Add all metric derivatives. */
    DerivativeType tmpDerivative = DerivativeType( this->GetNumberOfParameters() );
    derivative = DerivativeType( this->GetNumberOfParameters() );
    derivative.Fill( NumericTraits< MeasureType >::Zero );
    for ( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
    {
      tmpDerivative.Fill( NumericTraits< MeasureType >::Zero );
      this->m_Metrics[ i ]->GetDerivative( parameters, tmpDerivative );
      derivative += this->m_MetricWeights[ i ] * tmpDerivative;
    }

  } // end GetDerivative


  /**
	 * ********************* GetValueAndDerivative ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    CombinedImageToImageMetric<TFixedImage,TMovingImage>
    ::GetValueAndDerivative( const ParametersType & parameters,
      MeasureType & value, DerivativeType & derivative ) const
  {
    /** Initialise. */
    MeasureType tmpValue = NumericTraits< MeasureType >::Zero;
    DerivativeType tmpDerivative = DerivativeType( this->GetNumberOfParameters() );

    value = NumericTraits< MeasureType >::Zero;
    derivative = DerivativeType( this->GetNumberOfParameters() );
    derivative.Fill( NumericTraits< MeasureType >::Zero );

    /** Add all metric values and derivatives. */
    for ( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
    {
      tmpValue = NumericTraits< MeasureType >::Zero;
      tmpDerivative.Fill( NumericTraits< MeasureType >::Zero );
      this->m_Metrics[ i ]->GetValueAndDerivative( parameters, tmpValue, tmpDerivative );
      this->m_MetricValues[ i ] = this->m_MetricWeights[ i ] * tmpValue;
      value += this->m_MetricValues[ i ];
      derivative += this->m_MetricWeights[ i ] * tmpDerivative;
    }

  } // end GetValueAndDerivative


} // end namespace itk


#endif // end #ifndef _itkCombinedImageToImageMetric_txx

