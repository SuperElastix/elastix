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

#ifndef __itkEulerStackTransformInitializer_hxx
#define __itkEulerStackTransformInitializer_hxx

#include "itkEulerStackTransformInitializer.h"
#include "itkImageMaskSpatialObject.h"

namespace itk
{

/**
 * ************************* Constructor *********************
 */

template< class TTransform, class TRTransform, class TFixedImage, class TMovingImage >
EulerStackTransformInitializer< TTransform, TRTransform, TFixedImage, TMovingImage >
::EulerStackTransformInitializer()
{
}


/**
 * ************************* InitializeTransform *********************
 */

template< class TTransform, class TRTransform, class TFixedImage, class TMovingImage >
void
EulerStackTransformInitializer< TTransform, TRTransform, TFixedImage, TMovingImage >
::InitializeTransform( bool centerGivenAsIndex, bool centerGivenAsPoint,ContinuousIndexType centerOfRotationIndex,InputPointType centerOfRotationPoint ) const
{
  // Sanity check
  if( !this->m_MovingImage )
  {
    itkExceptionMacro( "Moving Image has not been set" );
    return;
  }
  if( !this->m_Transform )
  {
    itkExceptionMacro( "Transform has not been set" );
    return;
  }

  // If images come from filters, then update those filters.
  if( this->m_MovingImage->GetSource() )
  {
    this->m_MovingImage->GetSource()->Update();
  }

  OutputVectorType translationVector;

  typedef ImageMaskSpatialObject< OutputSpaceDimension> MovingMaskSpatialObjectType;

    // Convert the masks to spatial objects
    typename MovingMaskSpatialObjectType::Pointer movingMaskAsSpatialObject = 0;
    if( this->m_MovingMask )
    {
      movingMaskAsSpatialObject = MovingMaskSpatialObjectType::New();
      movingMaskAsSpatialObject->SetImage( this->m_MovingMask );
    }

    typedef typename MovingImageType::IndexType IndexType;

    double m_M0;
    ReducedDimensionOutputVectorType m_Cg;
    ReducedDimensionOutputVectorType m_Cg_m;
    m_Cg_m.Fill(NumericTraits< typename OutputVectorType::ValueType >::Zero);
    
    typename MovingImageType::RegionType region = this->m_MovingImage->GetLargestPossibleRegion();

    for(unsigned int j =0; j < this->m_MovingImage->GetLargestPossibleRegion().GetSize()[ReducedOutputSpaceDimension]; j++)
    {
        region.SetIndex(ReducedOutputSpaceDimension, j);
        region.SetSize(ReducedOutputSpaceDimension, 1);
        
        IteratorType it(this->m_MovingImage, region );
        
        m_M0 = 0;
        m_Cg.Fill(NumericTraits< typename ReducedDimensionOutputVectorType::ValueType >::Zero);

        
        for (it.GoToBegin();!it.IsAtEnd(); ++it) {
            
            double value = it.Value();
            IndexType indexPosition = it.GetIndex();
            
            Point< double, OutputSpaceDimension> physicalPosition;
            this->m_MovingImage->TransformIndexToPhysicalPoint(indexPosition, physicalPosition);

            if ( movingMaskAsSpatialObject.IsNull()
                || movingMaskAsSpatialObject->IsInside(physicalPosition) )
            {
                m_M0 += value;
                for ( unsigned int i = 0; i < ReducedOutputSpaceDimension; i++ )
                {
                    m_Cg[i] += physicalPosition[i] * value;
                }

            }
        }
        // Throw an error if the total mass is zero
        if ( m_M0 == 0.0 )
        {
            itkExceptionMacro(<< "Compute(): Total Mass of the image was zero. Aborting here to prevent division by zero later on.");
        }
        // Normalize using the total mass
        for ( unsigned int i = 0; i < ReducedOutputSpaceDimension; i++ )
        {
            m_Cg[i] /= m_M0;
            m_Cg_m[i] += m_Cg[i] / static_cast<float>(this->m_MovingImage->GetLargestPossibleRegion().GetSize()[ReducedOutputSpaceDimension]);
        }
    }
    for(unsigned int j =0; j < this->m_MovingImage->GetLargestPossibleRegion().GetSize()[ReducedOutputSpaceDimension]; j++)
    {
        region.SetIndex(ReducedOutputSpaceDimension, j);
        region.SetSize(ReducedOutputSpaceDimension, 1);
        
        IteratorType it(this->m_MovingImage, region );
        
        m_M0 = 0;
        m_Cg.Fill(NumericTraits< typename ReducedDimensionOutputVectorType::ValueType >::Zero);
        
        
        for (it.GoToBegin();!it.IsAtEnd(); ++it) {
            
            double value = it.Value();
            IndexType indexPosition = it.GetIndex();
            
            Point< double, OutputSpaceDimension> physicalPosition;
            this->m_MovingImage->TransformIndexToPhysicalPoint(indexPosition, physicalPosition);
            
            if ( movingMaskAsSpatialObject.IsNull()
                || movingMaskAsSpatialObject->IsInside(physicalPosition) )
            {
                m_M0 += value;
                for ( unsigned int i = 0; i < ReducedOutputSpaceDimension; i++ )
                {
                    m_Cg[i] += physicalPosition[i] * value;
                }
                
            }
        }
        // Throw an error if the total mass is zero
        if ( m_M0 == 0.0 )
        {
            itkExceptionMacro(<< "Compute(): Total Mass of the image was zero. Aborting here to prevent division by zero later on.");
        }
        // Normalize using the total mass
        for ( unsigned int i = 0; i < ReducedOutputSpaceDimension; i++ )
        {
            m_Cg[i] /= m_M0;
            m_Cg[i] -= m_Cg_m[i];
        }
        ReducedDimensionTransformPointer dummyReducedDimensionTransform = ReducedDimensionTransformType::New();
        dummyReducedDimensionTransform->SetIdentity();
        dummyReducedDimensionTransform->SetOffset(m_Cg);
        
        ReducedDimensionContinuousIndexType redDimCenterOfRotationIndex;
        ReducedDimensionInputPointType redDimCenterOfRotationPoint;
        SizeType fixedImageSize = m_MovingImage->GetLargestPossibleRegion().GetSize();
        for(unsigned int k = 0; k < ReducedInputSpaceDimension; k++)
        {

            redDimCenterOfRotationIndex[ k ] = 0;
            redDimCenterOfRotationPoint[ k ] = 0.0;
        }
        /** Determine the center of rotation as the center of the image if no center was given */
        const bool centerGiven = centerGivenAsIndex || centerGivenAsPoint;
        if ( !centerGiven  )
        {
            /** Use center of image as default center of rotation */
            for(unsigned int k = 0; k < ReducedInputSpaceDimension; k++)
            {
                centerOfRotationIndex[ k ] = (fixedImageSize[ k ] - 1.0f) / 2.0f;
            }
            
            /** Convert from continuous index to physical point */
            m_MovingImage->TransformContinuousIndexToPhysicalPoint( centerOfRotationIndex, centerOfRotationPoint );
            
            for(unsigned int k = 0; k < ReducedInputSpaceDimension; k++)
            {
                redDimCenterOfRotationPoint[ k ] = redDimCenterOfRotationPoint[ k ];
            }
            
        }
        
        /** Transform center of rotation point to physical point if given as index in parameter file. */
        if( centerGivenAsIndex)
        {
             m_MovingImage->TransformContinuousIndexToPhysicalPoint(centerOfRotationIndex, centerOfRotationPoint );
            
            for(unsigned int k = 0; k < ReducedInputSpaceDimension; k++)
            {
                redDimCenterOfRotationPoint[ k ] = centerOfRotationPoint[ k ];
            }
        }
        if(centerGivenAsPoint)
        {
            for(unsigned int k = 0; k < ReducedInputSpaceDimension; k++)
            {
                redDimCenterOfRotationPoint[ k ] = centerOfRotationPoint[ k ];
            }

        }
        
        /** Set the center of rotation point. */
        dummyReducedDimensionTransform->SetCenter( redDimCenterOfRotationPoint );
        
        this->m_Transform->SetSubTransform(j,dummyReducedDimensionTransform);

    }

} // end InitializeTransform()


/**
 * ************************* PrintSelf *********************
 */

template< class TTransform, class TRTransform, class TFixedImage, class TMovingImage >
void
EulerStackTransformInitializer< TTransform, TRTransform, TFixedImage, TMovingImage >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );

  os << indent << "Transform   = " << std::endl;
  if( this->m_Transform )
  {
    os << indent << this->m_Transform  << std::endl;
  }
  else
  {
    os << indent << "None" << std::endl;
  }

  os << indent << "MovingImage   = " << std::endl;
  if( this->m_MovingImage )
  {
    os << indent << this->m_MovingImage  << std::endl;
  }
  else
  {
    os << indent << "None" << std::endl;
  }

} // end PrintSelf()


}  // namespace itk

#endif /* __itkEulerStackTransformInitializer_hxx */
