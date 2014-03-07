/*======================================================================

  This file is part of the elastix software.

  Copyright (c) Erasmus MC, Rotterdam. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef ITKT1MAPPINGMETRIC_HXX
#define ITKT1MAPPINGMETRIC_HXX

#include <algorithm>

#include "itkT1MappingMetric.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "vnl/algo/vnl_matrix_update.h"
#include "vnl/vnl_inverse.h"
#include <numeric>
#include "LMOptimizer.h"
#include "T1MappingModel.h"

#define SIGN(a) ((a>0) - (a<0))
namespace itk
{
/**
 * ******************* Constructor *******************
 */

  template <class TFixedImage, class TMovingImage>
    T1MappingMetric<TFixedImage,TMovingImage>
      ::T1MappingMetric():
        m_TriggerTimes( 0.0 ),
        m_SubtractMean( false ),
        m_TransformIsStackTransform( false )
  {
    this->SetUseImageSampler( true );
    this->SetUseFixedImageLimiter( false );
    this->SetUseMovingImageLimiter( false );

  } // end constructor


/**
 * ******************* Initialize *******************
 */

  template <class TFixedImage, class TMovingImage>
    void
    T1MappingMetric<TFixedImage,TMovingImage>
    ::Initialize(void) throw ( ExceptionObject )
  {
    /** Initialize transform, interpolator, etc. */
    Superclass::Initialize();

    /** Update the imageSampler and get a handle to the sample container. */
    this->GetImageSampler()->Update();

    /** Retrieve slowest varying dimension and its size. */
    const unsigned int lastDim = this->GetFixedImage()->GetImageDimension() - 1;
    this->m_nrOfTimePoints = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );

   /** Resize TriggerTimes */
    VectorType dummy(this->m_nrOfTimePoints);

    for(unsigned int i = 0; i < this->m_nrOfTimePoints; ++i)
    {
        dummy[i] = this->m_TriggerTimes[i];
    }
    this->m_TriggerTimes = dummy;

    } // end Initialize


/**
 * ******************* PrintSelf *******************
 */

  template < class TFixedImage, class TMovingImage>
    void
    T1MappingMetric<TFixedImage,TMovingImage>
    ::PrintSelf(std::ostream& os, Indent indent) const
  {
    Superclass::PrintSelf( os, indent );

  } // end PrintSelf


/**
 * *************** EvaluateTransformJacobianInnerProduct ****************
 */

  template < class TFixedImage, class TMovingImage >
    void
    T1MappingMetric<TFixedImage,TMovingImage>
    ::EvaluateTransformJacobianInnerProduct(
    const TransformJacobianType & jacobian,
    const MovingImageDerivativeType & movingImageDerivative,
    DerivativeType & imageJacobian ) const
  {
    typedef typename TransformJacobianType::const_iterator JacobianIteratorType;
    typedef typename DerivativeType::iterator              DerivativeIteratorType;
    JacobianIteratorType jac = jacobian.begin();
    imageJacobian.Fill( 0.0 );
    const unsigned int sizeImageJacobian = imageJacobian.GetSize();
    for ( unsigned int dim = 0; dim < FixedImageDimension; dim++ )
    {
      const double imDeriv = movingImageDerivative[ dim ];
      DerivativeIteratorType imjac = imageJacobian.begin();

      for ( unsigned int mu = 0; mu < sizeImageJacobian; mu++ )
      {
        (*imjac) += (*jac) * imDeriv;
        ++imjac;
        ++jac;
      }
    }
  } // end EvaluateTransformJacobianInnerProduct


/**
 * ******************* GetValue *******************
 */

template <class TFixedImage, class TMovingImage>
    typename T1MappingMetric<TFixedImage,TMovingImage>::MeasureType
    T1MappingMetric<TFixedImage,TMovingImage>
    ::GetValue( const TransformParametersType & parameters ) const
  {
    itkDebugMacro( "GetValue( " << parameters << " ) " );

    /** Initialize some variables */
    this->m_NumberOfPixelsCounted = 0;
    MeasureType measure = NumericTraits< MeasureType >::Zero;

    /** Make sure the transform parameters are up to date. */
    this->SetTransformParameters( parameters );

    /** Initialize some variables */
    this->m_NumberOfPixelsCounted = 0;

    /** Update the imageSampler and get a handle to the sample container. */
    this->GetImageSampler()->Update();
    ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

    /** Create iterator over the sample container. */
    typename ImageSampleContainerType::ConstIterator fiter;
    typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
    typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

    /** Retrieve slowest varying dimension and its size. */
    const unsigned int lastDim = this->GetFixedImage()->GetImageDimension() - 1;

    unsigned int countminind = 0;
    unsigned int pixelIndex = 0;
     for ( fiter = fbegin; fiter != fend; ++fiter )
    {
        /** Initialize some variables. */
        RealType movingImageValue;
        MovingImagePointType mappedPoint;
        VectorType alpha( this->m_nrOfTimePoints );
        VectorType flippedVals( this->m_nrOfTimePoints);

      /** Read fixed coordinates. */
      FixedImagePointType fixedPoint = (*fiter).Value().m_ImageCoordinates;

      /** Transform sampled point to voxel coordinates. */
      FixedImageContinuousIndexType voxelCoord;
      this->GetFixedImage()->TransformPhysicalPointToContinuousIndex( fixedPoint, voxelCoord );

      for ( unsigned int timeindex = 0; timeindex < this->m_nrOfTimePoints; ++timeindex )
      {
          voxelCoord[ lastDim ] = timeindex;
          this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint( voxelCoord, fixedPoint );
          this->TransformPoint( fixedPoint, mappedPoint );
          this->EvaluateMovingImageValueAndDerivative(
                      mappedPoint, movingImageValue, 0 );
          alpha[timeindex] = movingImageValue;          
      }

      Sort( alpha );

      unsigned int numSamplesOK = 0;

     /////////////////////////////////////
     LMOptimizer::VectorType errvec( this->m_nrOfTimePoints );
     LMOptimizer * LM0 = new LMOptimizer;
     LM0->SetModel( new T1MappingModel() );
     for(unsigned int flipind = 0 ; flipind < this->m_nrOfTimePoints; flipind++ )
     {
         flippedVals = Flip( alpha, flipind );
         VectorType params = InitializeParams( flippedVals );
         LM0->SetValues( this->m_TriggerTimes, flippedVals );
         LM0->Run(this->m_NumberOfIterationsForLM, params );
         errvec[ flipind ] = ( LM0->GetErr() );
     }
     delete LM0;

     //  find index of minimum error
     double minvalue = *std::min_element(errvec.begin(), errvec.end()-1);
     unsigned int minindex=0;

     for( unsigned int ii = 0; ii < errvec.size();  ++ii)
     {
         if(errvec[ ii ] == minvalue )
         {
             minindex = ii;
         }
     }
////////////////////////////////////////

     LMOptimizer * LM = new LMOptimizer;
     LM->SetModel( new T1MappingModel() );
     flippedVals = Flip( alpha, minindex );
     VectorType params = InitializeParams( flippedVals );
     LM->SetValues( this->m_TriggerTimes, flippedVals );
     LM->Run( this->m_NumberOfIterationsForLM, params );

      for(unsigned int timeindex = 0; timeindex < this->m_nrOfTimePoints; ++timeindex)
      {
          voxelCoord[ lastDim ] = timeindex;
          this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint( voxelCoord, fixedPoint );
          bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );
          if ( sampleOk )
          {
              sampleOk = this->IsInsideMovingMask( mappedPoint );
          }
          if ( sampleOk )
          {
              sampleOk = this->EvaluateMovingImageValueAndDerivative( mappedPoint, movingImageValue, 0 );
          }
          if(sampleOk)
          {
              RealType diff = (LM->GetValues()[ timeindex ] - flippedVals[ timeindex ]);
              measure += diff * diff;
              ++numSamplesOK;
          }
      }
      ++pixelIndex;

      if( numSamplesOK > 0)
      {
          this->m_NumberOfPixelsCounted++;
      }
      delete LM;

    }
     /** Check if enough samples were valid. */
     this->CheckNumberOfSamples( sampleContainer->Size(), this->m_NumberOfPixelsCounted );

    double normal_sum = NumericTraits< MeasureType >::Zero;
    normal_sum = 1.0/static_cast<double>( this->m_NumberOfPixelsCounted*this->m_nrOfTimePoints);
     measure *= normal_sum;

    /** Return the mean squares measure value. */
    return measure;

  } // end GetValue


/**
 * ******************* GetDerivative *******************
 */

template < class TFixedImage, class TMovingImage>
    void
    T1MappingMetric<TFixedImage,TMovingImage>
    ::GetDerivative( const TransformParametersType & parameters,
    DerivativeType & derivative ) const
  {
    /** When the derivative is calculated, all information for calculating
     * the metric value is available. It does not cost anything to calculate
     * the metric value now. Therefore, we have chosen to only implement the
     * GetValueAndDerivative(), supplying it with a dummy value variable. */
    MeasureType dummyvalue = NumericTraits< MeasureType >::Zero;
    this->GetValueAndDerivative( parameters, dummyvalue, derivative );

  } // end GetDerivative


/**
     * ******************* GetValueAndDerivative *******************
     */

    template <class TFixedImage, class TMovingImage>
    void
    T1MappingMetric<TFixedImage,TMovingImage>
    ::GetValueAndDerivative( const TransformParametersType & parameters,
                             MeasureType & value, DerivativeType & derivative ) const
    {
        itkDebugMacro("GetValueAndDerivative( " << parameters << " ) ");
        /** Define derivative and Jacobian types. */
        typedef typename TransformJacobianType::ValueType TransformJacobianValueType;

        /** Initialize some variables */
        MeasureType measure = NumericTraits< MeasureType >::Zero;
        derivative = DerivativeType( this->GetNumberOfParameters() );
        derivative.Fill( NumericTraits< DerivativeValueType >::Zero );

        /** Array that stores dM(x)/dmu, and the sparse jacobian+indices. */
        NonZeroJacobianIndicesType nzji(
                    this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
        DerivativeType imageJacobian( nzji.size() );
        TransformJacobianType jacobian;

        /** Make sure the transform parameters are up to date. */
        this->SetTransformParameters( parameters );

        /** Update the imageSampler and get a handle to the sample container. */
        this->GetImageSampler()->Update();
        ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

        /** Create iterator over the sample container. */
        typename ImageSampleContainerType::ConstIterator fiter;
        typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
        typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

        /** Retrieve slowest varying dimension and its size. */
        const unsigned int lastDim = this->GetFixedImage()->GetImageDimension() - 1;

        /** Initialize some variables */
        unsigned int pixelIndex = 0;
        this->m_NumberOfPixelsCounted = 0;

        VectorType alpha(this->m_nrOfTimePoints);

        for ( fiter = fbegin; fiter != fend; ++fiter )
        {
            /** Initialize some variables. */
            RealType movingImageValue;
            MovingImageDerivativeType movingImageDerivative;
            MovingImagePointType mappedPoint;

            /** Read fixed coordinates. */
            FixedImagePointType fixedPoint = (*fiter).Value().m_ImageCoordinates;

            /** Transform sampled point to voxel coordinates. */
            FixedImageContinuousIndexType voxelCoord;
            this->GetFixedImage()->TransformPhysicalPointToContinuousIndex( fixedPoint, voxelCoord );

            for ( unsigned int timeindex = 0; timeindex < this->m_nrOfTimePoints; ++timeindex )
            {
                voxelCoord[ lastDim ] = timeindex;
                this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint( voxelCoord, fixedPoint );
                this->TransformPoint( fixedPoint, mappedPoint );
                this->EvaluateMovingImageValueAndDerivative( mappedPoint, movingImageValue, 0 );
                 alpha[timeindex] = movingImageValue;
            }

            // Sort the values of a pixel over times //
            Sort(alpha);

            unsigned int numSamplesOK = 0;

            /////////////////////////////////////
            LMOptimizer::VectorType errvec( this->m_nrOfTimePoints-5 );
            LMOptimizer * LM0 = new LMOptimizer;
            LM0->SetModel( new T1MappingModel() );
            for(unsigned int flipind = 0 ; flipind < this->m_nrOfTimePoints-5; flipind++ )
            {
                VectorType flippedVals0 = Flip( alpha, flipind );
                VectorType params0 = InitializeParams( flippedVals0 );
                LM0->SetValues( this->m_TriggerTimes, flippedVals0 );
                LM0->Run(this->m_NumberOfIterationsForLM, params0 );
                errvec[ flipind ] = ( LM0->GetErr() );
            }
            delete LM0;

            //  find index of minimum error
            double minvalue = *std::min_element(errvec.begin(), errvec.end());
            unsigned int minindex;

            for( unsigned int i = 0; i < errvec.size();  ++i)
            {
                if(errvec[ i ] == minvalue )
                {
                    minindex = i;
                }
            }
       ////////////////////////////////////////
            LMOptimizer * LM = new LMOptimizer;
            LM->SetModel( new T1MappingModel() );
            VectorType flippedVals = Flip( alpha, minindex );
            VectorType params = InitializeParams( flippedVals );
            LM->SetValues( this->m_TriggerTimes, flippedVals );
            LM->Run( m_NumberOfIterationsForLM, params );

            /** loop over time 2 */
            for(unsigned int timeindex = 0; timeindex < this->m_nrOfTimePoints; ++timeindex)
            {
                voxelCoord[ lastDim ] = timeindex;
                this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint( voxelCoord, fixedPoint );
                bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );
                if ( sampleOk )
                {
                    sampleOk = this->IsInsideMovingMask( mappedPoint );
                }
                if ( sampleOk )
                {
                    sampleOk = this->EvaluateMovingImageValueAndDerivative( mappedPoint, movingImageValue, &movingImageDerivative );
                }

                if( sampleOk )
                {
                    RealType diff = (LM->GetValues()[ timeindex ] - flippedVals[ timeindex ]);
                    this->EvaluateTransformJacobian( fixedPoint, jacobian, nzji );
                    this->EvaluateTransformJacobianInnerProduct( jacobian, movingImageDerivative, imageJacobian );
                    this->UpdateValueAndDerivativeTerms(SIGN(flippedVals[ timeindex ]), diff, imageJacobian, nzji, measure, derivative);
                    ++numSamplesOK;
                }
            }

            if( numSamplesOK > 0 )
            {
                this->m_NumberOfPixelsCounted++;
                ++pixelIndex;
            }
            delete LM;
        }

        /** Check if enough samples were valid. */
        this->CheckNumberOfSamples( sampleContainer->Size(), this->m_NumberOfPixelsCounted );

        /** Subtract mean from derivative elements. */
        if( this->m_SubtractMean)
        {
            if ( ! this->m_TransformIsStackTransform )
            {
                /** Update derivative per dimension.
                  * Parameters are ordered xxxxxxx yyyyyyy zzzzzzz ttttttt and
                  * per dimension xyz.
                  */

                const unsigned int lastDimGridSize = this->m_GridSize[ lastDim ];
                const unsigned int numParametersPerDimension
                        = this->GetNumberOfParameters() / this->GetMovingImage()->GetImageDimension();
                const unsigned int numControlPointsPerDimension = numParametersPerDimension / lastDimGridSize;
                DerivativeType mean ( numControlPointsPerDimension );
                for ( unsigned int d = 0; d < this->GetMovingImage()->GetImageDimension(); ++d )
                {
                    /** Compute mean per dimension. */
                    mean.Fill( 0.0 );
                    const unsigned int starti = numParametersPerDimension * d;
                    for ( unsigned int i = starti; i < starti + numParametersPerDimension; ++i )
                    {
                        const unsigned int index = i % numControlPointsPerDimension;
                        mean[ index ] += derivative[ i ];
                    }
                    mean /= static_cast< double >( lastDimGridSize );

                    /** Update derivative for every control point per dimension. */
                    for ( unsigned int i = starti; i < starti + numParametersPerDimension; ++i )
                    {
                        const unsigned int index = i % numControlPointsPerDimension;
                        derivative[ i ] -= mean[ index ];
                    }
                }
            }
            else
            {
                const unsigned int numParametersPerLastDimension = this->GetNumberOfParameters() / this->m_nrOfTimePoints;
                DerivativeType mean ( numParametersPerLastDimension );
                mean.Fill( 0.0 );

                /** Compute mean per control point. */
                for ( unsigned int t = 0; t < this->m_nrOfTimePoints; ++t )
                {
                    const unsigned int startc = numParametersPerLastDimension * t;
                    for ( unsigned int c = startc; c < startc + numParametersPerLastDimension; ++c )
                    {
                        const unsigned int index = c % numParametersPerLastDimension;
                        mean[ index ] += derivative[ c ];
                    }
                }
                mean /= static_cast< double >( this->m_nrOfTimePoints );
                /** Update derivative per control point. */
                for ( unsigned int t = 0; t < this->m_nrOfTimePoints; ++t )
                {
                    const unsigned int startc = numParametersPerLastDimension * t;
                    for ( unsigned int c = startc; c < startc + numParametersPerLastDimension; ++c )
                    {
                        const unsigned int index = c % numParametersPerLastDimension;
                        derivative[ c ] -= mean[ index ];
                    }
                }
            }
        }

        /** Compute the measure value and derivative. */
        double normal_sum = 0.0;
        normal_sum = 1.0/static_cast<double>( this->m_NumberOfPixelsCounted *this->m_nrOfTimePoints);
        measure *= normal_sum;
        derivative *= normal_sum;
        value = measure;

    } // end GetValueAndDerivative()

    /**
  * *************** UpdateValueAndDerivativeTerms ***************************
  */

    template < class TFixedImage, class TMovingImage >
    void
    T1MappingMetric<TFixedImage,TMovingImage>
    ::UpdateValueAndDerivativeTerms(int sign,
            const RealType  & diff,
            const DerivativeType & imageJacobian,
            const NonZeroJacobianIndicesType & nzji,
            MeasureType & measure,
            DerivativeType & deriv) const
    {
        /** The difference squared. */

         measure += (diff * diff);

        /** Calculate the contributions to the derivatives with respect to each parameter. */
        const RealType diff_2 = diff * 2.0;

        /** Only pick the nonzero Jacobians. */
        for ( unsigned int i = 0; i < imageJacobian.GetSize(); ++i )
        {
            const unsigned int index = nzji[ i ];

            //Assume dSdmu = 0
            deriv[ index ] += (diff_2 *(  -sign*imageJacobian[ i ] ));
        }
    } // end UpdateValueAndDerivativeTerms

    /**
      * ******************* Sort Values *******************
      */
    template < class TFixedImage, class TMovingImage >
    void
    T1MappingMetric<TFixedImage,TMovingImage>
    ::Sort( VectorType & alpha ) const
   {
       VectorType values( this->m_nrOfTimePoints );
        std::vector< mypair > pairvec(this->m_nrOfTimePoints);
        for (unsigned int i = 0; i < this->m_nrOfTimePoints; ++i)
        {
            mypair temppair = std::make_pair(this->m_TriggerTimes[ i ], i );
            pairvec[ i ] = temppair;
        }

        std::sort(pairvec.begin(), pairvec.end());
        VectorType index_order( this->m_nrOfTimePoints);
        for(unsigned int i = 0; i < this->m_nrOfTimePoints; ++i)
        {
            mypair t = pairvec[ i ];
            index_order[ i ] = t.second;
            values[ index_order[ i ] ] = alpha[ i ];
        }
        alpha = values;
    }


    /**
      * ******************* Flip time points *******************
      */
    template < class TFixedImage, class TMovingImage >
    typename T1MappingMetric<TFixedImage,TMovingImage>::VectorType
    T1MappingMetric<TFixedImage,TMovingImage>
    ::Flip( const VectorType & alpha, const unsigned int flipind ) const
    {
        VectorType flipped( this->m_nrOfTimePoints );
        flipped = alpha;

        for( unsigned int i = 0; i <= flipind;  ++i)
        {
           flipped[ i ] = -alpha[ i ];
        }

        return flipped;
    }

    /**
      * ******************* InitializeModelParameters*******************
      */
    template <class TFixedImage, class TMovingImage>
    typename T1MappingMetric<TFixedImage,TMovingImage>::VectorType
    T1MappingMetric<TFixedImage,TMovingImage>
    ::InitializeParams ( const VectorType & alpha ) const
    {
        VectorType p( 3 );

        p[ 2 ] =  *max_element(alpha.begin(), alpha.end())+1.0; //C3 is the value belonging to the latest point in time
        p[ 0 ] = -2.0*p[ 2 ] ;

        vnl_vector< double > C2vec( this->m_nrOfTimePoints );
        std::fill(C2vec.begin(), C2vec.end(),  0.0 );

        VectorType timeSorted( this->m_nrOfTimePoints );
        timeSorted = this->m_TriggerTimes;
        std::sort(timeSorted.begin(), timeSorted.end());

        for( unsigned int i = 0; i < this->m_nrOfTimePoints;  ++i)
        {
            C2vec[ i ] = -log((alpha[ i ] - p[ 2 ])/p[ 0 ])/timeSorted[ i ];
        }

        // First calculate the median of C2vec
        double* medIter = NULL;
        medIter = C2vec.begin() + this->m_nrOfTimePoints/2;
        std::nth_element(C2vec.begin(), medIter, C2vec.end());
        p[ 1 ] = *medIter;

        return p;
    }
} // end namespace itk

#endif // ITKT1MAPPINGMETRIC_HXX

