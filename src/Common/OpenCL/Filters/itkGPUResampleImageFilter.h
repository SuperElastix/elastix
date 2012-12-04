/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUResampleImageFilter_h
#define __itkGPUResampleImageFilter_h

#include "itkResampleImageFilter.h"

#include "itkGPUImageToImageFilter.h"
#include "itkGPUInterpolateImageFunction.h"
#include "itkGPUTransformBase.h"

#include "itkVersion.h"
#include "itkObjectFactoryBase.h"

#include <vector>
#include <string>
#include <utility>

namespace itk
{
/** \class GPUResampleImageFilter
* \ingroup ITK-GPUCommon
*/

/** Create a helper GPU Kernel class for GPUResampleImageFilter */
itkGPUKernelClassMacro( GPUResampleImageFilterKernel );

template< class TInputImage, class TOutputImage, class TInterpolatorPrecisionType = float >
class ITK_EXPORT GPUResampleImageFilter :
  public GPUImageToImageFilter< TInputImage, TOutputImage,
                                ResampleImageFilter< TInputImage, TOutputImage, TInterpolatorPrecisionType > >
{
public:
  /** Standard class typedefs. */
  typedef GPUResampleImageFilter                                                       Self;
  typedef ResampleImageFilter< TInputImage, TOutputImage, TInterpolatorPrecisionType > CPUSuperclass;
  typedef GPUImageToImageFilter< TInputImage, TOutputImage, CPUSuperclass >            GPUSuperclass;
  typedef SmartPointer< Self >                                                         Pointer;
  typedef SmartPointer< const Self >                                                   ConstPointer;

  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUResampleImageFilter, GPUSuperclass );

  /** Superclass typedefs. */
  typedef typename GPUSuperclass::OutputImageRegionType OutputImageRegionType;
  typedef typename GPUSuperclass::OutputImagePixelType  OutputImagePixelType;

  /** Some convenient typedefs. */
  typedef TInputImage                              InputImageType;
  typedef TOutputImage                             OutputImageType;
  typedef typename GPUTraits< TInputImage >::Type  GPUInputImage;
  typedef typename GPUTraits< TOutputImage >::Type GPUOutputImage;
  typedef typename CPUSuperclass::InterpolatorType InterpolatorType;
  typedef typename CPUSuperclass::TransformType    TransformType;
  typedef typename OutputImageType::IndexType      IndexType;

  /** Scheduler typedefs. */
  typedef typename GPUKernelManager::Pointer GPUKernelManagerPointer;

  /** ImageDimension constants */
  itkStaticConstMacro( InputImageDimension, unsigned int,
                       TInputImage::ImageDimension );
  itkStaticConstMacro( OutputImageDimension, unsigned int,
                       TOutputImage::ImageDimension );

  virtual void SetInterpolator( InterpolatorType *_arg );

  virtual void SetTransform( const TransformType *_arg );

protected:
  GPUResampleImageFilter();
  ~GPUResampleImageFilter() {}
  virtual void PrintSelf( std::ostream & os, Indent indent ) const;

  virtual void GPUGenerateData();

  // Supported GPU transform types
  typedef enum {
    IdentityTransform = 1,
    MatrixOffsetTransform,
    TranslationTransform,
    BSplineTransform,
    Else
  } GPUInputTransformType;

  void SetArgumentsForPreKernelManager(
    const typename GPUOutputImage::Pointer & output,
    cl_uint & index );

  void SetArgumentsForLoopKernelManager(
    const typename GPUInputImage::Pointer & input,
    const typename GPUOutputImage::Pointer & output,
    cl_uint & tsizeLoopIntex,
    cl_uint & comboIntex,
    cl_uint & transformIndex );

  void SetTransformArgumentsForLoopKernelManager(
    const std::size_t index,
    const cl_uint comboIndex,
    const cl_uint transformIndex );

  void SetArgumentsForPostKernelManager(
    const typename GPUOutputImage::Pointer & input,
    const typename GPUOutputImage::Pointer & output,
    cl_uint & index );

  void SetGPUCoefficients(
    const std::size_t index, const cl_uint transformindex );

  bool HasTransform( const GPUInputTransformType type );

  int  GetTransformHandle( const GPUInputTransformType type );

  void CalculateDelta(
    const typename GPUInputImage::Pointer & _inputPtr,
    const typename GPUOutputImage::Pointer & _outputPtr,
    float *_delta );

private:
  GPUResampleImageFilter( const Self & ); // purposely not implemented
  void operator=( const Self & );         // purposely not implemented

  GPUInterpolatorBase *m_InterpolatorBase;
  GPUTransformBase *   m_TransformBase;

  typename GPUDataManager::Pointer m_InputGPUImageBase;
  typename GPUDataManager::Pointer m_OutputGPUImageBase;
  typename GPUDataManager::Pointer m_Parameters;
  typename GPUDataManager::Pointer m_DeformationFieldBuffer;

  typedef std::pair< int, bool >                             TransformHandle;
  typedef std::map< GPUInputTransformType, TransformHandle > TransformsHandle;

  std::vector< std::string > m_Sources;
  std::size_t                m_SourceIndex;

  std::size_t m_InterpolatorSourceLoadedIndex;
  std::size_t m_TransformSourceLoadedIndex;

  bool m_InterpolatorIsBSpline;
  bool m_TransformIsCombo;

  int              m_FilterPreGPUKernelHandle;
  TransformsHandle m_FilterLoopGPUKernelHandle;
  int              m_FilterPostGPUKernelHandle;

  // GPU kernel managers
  GPUKernelManagerPointer m_PreKernelManager;
  GPUKernelManagerPointer m_LoopKernelManager;
  GPUKernelManagerPointer m_PostKernelManager;
};

/** \class GPUResampleImageFilterFactory
* \brief Object Factory implementation for GPUResampleImageFilter
*/
class GPUResampleImageFilterFactory : public ObjectFactoryBase
{
public:
  typedef GPUResampleImageFilterFactory Self;
  typedef ObjectFactoryBase             Superclass;
  typedef SmartPointer< Self >          Pointer;
  typedef SmartPointer< const Self >    ConstPointer;

  /** Class methods used to interface with the registered factories. */
  virtual const char * GetITKSourceVersion() const { return ITK_SOURCE_VERSION; }
  const char * GetDescription() const { return "A Factory for GPUResampleImageFilter"; }

  /** Method for class instantiation. */
  itkFactorylessNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUResampleImageFilterFactory, itk::ObjectFactoryBase );

  /** Register one factory of this type  */
  static void RegisterOneFactory( void )
  {
    GPUResampleImageFilterFactory::Pointer factory =
      GPUResampleImageFilterFactory::New();
    ObjectFactoryBase::RegisterFactory( factory );
  }

private:
  GPUResampleImageFilterFactory( const Self & ); // purposely not implemented
  void operator=( const Self & );                // purposely not implemented

#define OverrideResampleImageFilterTypeMacro( ipt, opt, pt, dm )                                      \
  {                                                                                                   \
    typedef Image< ipt, dm > InputImageType;                                                          \
    typedef Image< opt, dm > OutputImageType;                                                         \
    this->RegisterOverride(                                                                           \
      typeid( ResampleImageFilter< InputImageType, OutputImageType, pt > ).name(),                    \
      typeid( GPUResampleImageFilter< InputImageType, OutputImageType, pt > ).name(),                 \
      "GPU ResampleImageFilter Override",                                                             \
      true,                                                                                           \
      CreateObjectFunction< GPUResampleImageFilter< InputImageType, OutputImageType, pt > >::New() ); \
  }

  GPUResampleImageFilterFactory()
  {
    if ( IsGPUAvailable() )
    {
      // general types
      //OverrideResampleImageFilterTypeMacro(unsigned char, unsigned char,
      // float, 1);
      //OverrideResampleImageFilterTypeMacro(char, char, float, 1);
      //OverrideResampleImageFilterTypeMacro(unsigned short, unsigned short,
      // float, 1);
      OverrideResampleImageFilterTypeMacro( short, short, float, 1 );
      //OverrideResampleImageFilterTypeMacro(unsigned int, unsigned int, float,
      // 1);
      //OverrideResampleImageFilterTypeMacro(int, int, float, 1);
      //OverrideResampleImageFilterTypeMacro(float, float, float, 1);
      //OverrideResampleImageFilterTypeMacro(double, double, float, 1);

      //OverrideResampleImageFilterTypeMacro(unsigned char, unsigned char,
      // float, 2);
      //OverrideResampleImageFilterTypeMacro(char, char, float, 2);
      //OverrideResampleImageFilterTypeMacro(unsigned short, unsigned short,
      // float, 2);
      OverrideResampleImageFilterTypeMacro( short, short, float, 2 );
      //OverrideResampleImageFilterTypeMacro(unsigned int, unsigned int, float,
      // 2);
      //OverrideResampleImageFilterTypeMacro(int, int, float, 2);
      //OverrideResampleImageFilterTypeMacro(float, float, float, 2);
      //OverrideResampleImageFilterTypeMacro(double, double, float, 2);

      //OverrideResampleImageFilterTypeMacro(unsigned char, unsigned char,
      // float, 3);
      //OverrideResampleImageFilterTypeMacro(char, char, float, 3);
      //OverrideResampleImageFilterTypeMacro(unsigned short, unsigned short,
      // float, 3);
      OverrideResampleImageFilterTypeMacro( short, short, float, 3 );
      //OverrideResampleImageFilterTypeMacro(unsigned int, unsigned int, float,
      // 3);
      //OverrideResampleImageFilterTypeMacro(int, int, float, 3);
      //OverrideResampleImageFilterTypeMacro(float, float, float, 3);
      //OverrideResampleImageFilterTypeMacro(double, double, float, 3);

      // type to float
      //OverrideResampleImageFilterTypeMacro(unsigned char, float, float, 1);
      //OverrideResampleImageFilterTypeMacro(char, float, float, 1);
      //OverrideResampleImageFilterTypeMacro(unsigned short, float, float, 1);
      //OverrideResampleImageFilterTypeMacro(short, float, float, 1);
      //OverrideResampleImageFilterTypeMacro(unsigned int, float, float, 1);
      //OverrideResampleImageFilterTypeMacro(int, float, float, 1);

      //OverrideResampleImageFilterTypeMacro(unsigned char, float, float, 2);
      //OverrideResampleImageFilterTypeMacro(char, float, float, 2);
      //OverrideResampleImageFilterTypeMacro(unsigned short, float, float, 2);
      //OverrideResampleImageFilterTypeMacro(short, float, float, 2);
      //OverrideResampleImageFilterTypeMacro(unsigned int, float, float, 2);
      //OverrideResampleImageFilterTypeMacro(int, float, float, 2);

      //OverrideResampleImageFilterTypeMacro(unsigned char, float, float, 3);
      //OverrideResampleImageFilterTypeMacro(char, float, float, 3);
      //OverrideResampleImageFilterTypeMacro(unsigned short, float, float, 3);
      //OverrideResampleImageFilterTypeMacro(short, float, float, 3);
      //OverrideResampleImageFilterTypeMacro(unsigned int, float, float, 3);
      //OverrideResampleImageFilterTypeMacro(int, float, float, 3);
    }
  }
};
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGPUResampleImageFilter.hxx"
#endif

#endif /* __itkGPUResampleImageFilter_h */
