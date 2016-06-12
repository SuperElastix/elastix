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

#ifndef __itkTranslationStackTransformInitializer_hxx
#define __itkTranslationStackTransformInitializer_hxx

#include "itkTranslationStackTransformInitializer.h"
#include "itkImageMaskSpatialObject.h"

namespace itk
{

/**
 * ************************* Constructor *********************
 */

template< class TTransform, class TRTransform, class TFixedImage, class TMovingImage >
TranslationStackTransformInitializer< TTransform, TRTransform, TFixedImage, TMovingImage >
::TranslationStackTransformInitializer()
{
}


/**
 * ************************* InitializeTransform *********************
 */

template< class TTransform, class TRTransform, class TFixedImage, class TMovingImage >
void
TranslationStackTransformInitializer< TTransform, TRTransform, TFixedImage, TMovingImage >
::InitializeTransform( void ) const
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
        dummyReducedDimensionTransform->SetOffset(m_Cg);
        this->m_Transform->SetSubTransform(j,dummyReducedDimensionTransform);

    }

} // end InitializeTransform()


/**
 * ************************* PrintSelf *********************
 */

template< class TTransform, class TRTransform, class TFixedImage, class TMovingImage >
void
TranslationStackTransformInitializer< TTransform, TRTransform, TFixedImage, TMovingImage >
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

#endif /* __itkTranslationStackTransformInitializer_hxx */
