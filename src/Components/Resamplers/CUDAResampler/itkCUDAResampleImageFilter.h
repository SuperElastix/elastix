/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkCUDAResamplerImageFilter_h
#define __itkCUDAResamplerImageFilter_h

#include "itkImage.h"
#include "itkResampleImageFilter.h"
#include "itkAdvancedCombinationTransform.h"
#include "itkAdvancedBSplineDeformableTransform.h"
#include "itkBSplineDeformableTransform.h"
#include "cudaResampleImageFilter.cuh"

namespace itk
{

/** \class itkCUDAResampleImageFilter
 * \brief Resample an image on the GPU via a coordinate transform.
 *
 * This class is an ITK wrap around a pure CUDA resampling class.
 *
 * \warning The implementation is currently very limited: only
 * a single third order B-spline transform is supported for 3D
 * images together with third order B-spline interpolation.
 *
 * \ingroup GeometricTransforms
 */

template< typename TInputImage, typename TOutputImage, typename TInterpolatorPrecisionType = float >
class ITK_EXPORT itkCUDAResampleImageFilter :
  public         ResampleImageFilter< TInputImage, TOutputImage, TInterpolatorPrecisionType >
{
public:

  /** Standard class typedefs. */
  typedef itkCUDAResampleImageFilter Self;
  typedef ResampleImageFilter<
    TInputImage, TOutputImage, TInterpolatorPrecisionType >      Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( itkCUDAResampleImageFilter, ResampleImageFilter );

  /** Typedefs from Superclass. */
  typedef typename Superclass::InputImageType         InputImageType;
  typedef typename Superclass::OutputImageType        OutputImageType;
  typedef typename Superclass::InputImagePointer      InputImagePointer;
  typedef typename Superclass::InputImageConstPointer InputImageConstPointer;
  typedef typename Superclass::OutputImagePointer     OutputImagePointer;
  typedef typename Superclass::InputImageRegionType   InputImageRegionType;

  typedef typename Superclass::TransformType           TransformType;
  typedef typename Superclass::TransformPointerType    TransformPointerType;
  typedef typename Superclass::InterpolatorType        InterpolatorType;
  typedef typename Superclass::InterpolatorPointerType InterpolatorPointerType;

  typedef typename Superclass::SizeType              SizeType;
  typedef typename Superclass::IndexType             IndexType;
  typedef typename Superclass::PointType             PointType;
  typedef typename Superclass::PixelType             PixelType;
  typedef typename Superclass::InputPixelType        InputPixelType;
  typedef typename Superclass::OutputImageRegionType OutputImageRegionType;
  typedef typename Superclass::SpacingType           SpacingType;
  typedef typename Superclass::OriginPointType       OriginPointType;
  typedef typename Superclass::DirectionType         DirectionType;
  typedef typename Superclass::ImageBaseType         ImageBaseType;

  /** Typedefs. */
  typedef AdvancedCombinationTransform<
    TInterpolatorPrecisionType, 3 >                     InternalComboTransformType;
  typedef AdvancedBSplineDeformableTransform<
    TInterpolatorPrecisionType, 3, 3 >                  InternalAdvancedBSplineTransformType;
  typedef typename InternalAdvancedBSplineTransformType::Pointer      ValidTransformPointer;
  typedef typename InternalAdvancedBSplineTransformType::ConstPointer ValidTransformConstPointer;
  typedef BSplineDeformableTransform<
    TInterpolatorPrecisionType, 3, 3 >                  InternalBSplineTransformType;
  typedef cuda::CUDAResampleImageFilter<
    typename InternalBSplineTransformType::ParametersValueType,
    typename TInputImage::PixelType, float >            CudaResampleImageFilterType;

  /** Set whether to use the GPU. */
  itkSetMacro( UseCuda, bool );
  itkGetConstMacro( UseCuda, bool );
  itkBooleanMacro( UseCuda );

  /** Set whether to use the GPU for casting (or the CPU). */
  itkSetMacro( UseGPUToCastData, bool );
  itkGetConstMacro( UseGPUToCastData, bool );
  itkBooleanMacro( UseGPUToCastData );

  /** Set whether to use the fast or the accurate CUDA kernel. */
  itkSetMacro( UseFastCUDAKernel, bool );
  itkGetConstMacro( UseFastCUDAKernel, bool );
  itkBooleanMacro( UseFastCUDAKernel );

  /** Implements GPU resampling. */
  virtual void GenerateData( void );

  /** For reporting warnings. */
  class WarningReportType
  {
public:

    std::vector< std::string > m_Warnings;

    void ResetWarningReport( void )
    {
      this->m_Warnings.resize( 0 );
    }


    std::string GetWarningReportAsString( void ) const
    {
      std::string warnings = "\n---------------------------------\n";
      for( std::size_t i = 0; i < this->m_Warnings.size(); i++ )
      {
        warnings += "itkCUDAResampleImageFilter: " + this->m_Warnings[ i ];
        warnings += "\n---------------------------------\n";
      }
      return warnings;
    }


  };

  //itkGetConstReferenceMacro( WarningReport, WarningReportType );
  virtual const WarningReportType & GetWarningReport( void ) const
  {
    return this->m_WarningReport;
  }


protected:

  itkCUDAResampleImageFilter();
  ~itkCUDAResampleImageFilter();

  virtual void CheckForValidConfiguration( ValidTransformPointer & bSplineTransform );

private:

  /** Private members. */
  bool m_UseCuda;
  bool m_UseGPUToCastData;
  bool m_UseFastCUDAKernel;

  CudaResampleImageFilterType m_CudaResampleImageFilter;
  WarningReportType           m_WarningReport;

  /** Helper function to check for a valid transform.
   * Currently, only GPU resampling of 3D images for 3-rd order B-splines is supported,
   * and only for one transform, so no concatenations.
   */
  bool CheckForValidTransform( ValidTransformPointer & bSplineTransform ) const;

  /** Helper function to check for a valid interpolator.
   * Currently, only GPU resampling using 3-rd order B-spline interpolation is supported.
   */
  bool CheckForValidInterpolator( void ) const;

  /** Helper function to check for a valid direction cosines.
   * Currently, only GPU resampling using identity cosines is supported.
   */
  bool CheckForValidDirectionCosines( ValidTransformPointer bSplineTransform ); //const;

  // NOTE: const can be added again in ITK4. It's due to GetInput() being not const-correct.

  /** Helper function to copy data. */
  void CopyParameters( ValidTransformPointer bSplineTransform );

};

// end class itkCUDAResampleImageFilter

}  // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCUDAResampleImageFilter.hxx"
#endif

#endif // end #ifndef __itkCUDAResamplerImageFilter_h
