/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPURecursiveGaussianImageFilter_h
#define __itkGPURecursiveGaussianImageFilter_h

#include "itkRecursiveGaussianImageFilter.h"

#include "itkGPUInPlaceImageFilter.h"
#include "itkVersion.h"
#include "itkObjectFactoryBase.h"

namespace itk
{
/** \class GPURecursiveGaussianImageFilter
* \brief Base class for computing IIR convolution with an approximation of a Gaussian kernel.
*
*    \f[
*      \frac{ 1 }{ \sigma \sqrt{ 2 \pi } } \exp{ \left( - \frac{x^2}{ 2 \sigma^2 } \right) }
*    \f]
*
* RecursiveGaussianImageFilter is the base class for recursive filters that
* approximate convolution with the Gaussian kernel.
* This class implements the recursive filtering
* method proposed by R.Deriche in IEEE-PAMI
* Vol.12, No.1, January 1990, pp 78-87,
* "Fast Algorithms for Low-Level Vision"
*
* Details of the implementation are described in the technical report:
* R. Deriche, "Recursively Implementing The Gaussian and Its Derivatives",
* INRIA, 1993, ftp://ftp.inria.fr/INRIA/tech-reports/RR/RR-1893.ps.gz
*
* Further improvements of the algorithm are described in:
* G. Farneback & C.-F. Westin, "On Implementation of Recursive Gaussian
* Filters", so far unpublished.
*
* \see RecursiveGaussianImageFilter
* \ingroup ITK-GPUCommon
*/

/** Create a helper GPU Kernel class for GPURecursiveGaussianImageFilter */
itkGPUKernelClassMacro( GPURecursiveGaussianImageFilterKernel );

template< class TInputImage, class TOutputImage >
class ITK_EXPORT GPURecursiveGaussianImageFilter :
  public GPUInPlaceImageFilter< TInputImage, TOutputImage,
                                RecursiveGaussianImageFilter< TInputImage, TOutputImage > >
{
public:
  /** Standard class typedefs. */
  typedef GPURecursiveGaussianImageFilter                                   Self;
  typedef RecursiveGaussianImageFilter< TInputImage, TOutputImage >         CPUSuperclass;
  typedef GPUImageToImageFilter< TInputImage, TOutputImage, CPUSuperclass > GPUSuperclass;
  typedef SmartPointer< Self >                                              Pointer;
  typedef SmartPointer< const Self >                                        ConstPointer;

  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPURecursiveGaussianImageFilter, GPUSuperclass );

  /** Superclass typedefs. */
  typedef typename GPUSuperclass::OutputImageRegionType OutputImageRegionType;
  typedef typename GPUSuperclass::OutputImagePixelType  OutputImagePixelType;
  typedef typename CPUSuperclass::ScalarRealType        ScalarRealType;

  /** Some convenient typedefs. */
  typedef TInputImage                           InputImageType;
  typedef typename InputImageType::Pointer      InputImagePointer;
  typedef typename InputImageType::ConstPointer InputImageConstPointer;
  typedef typename InputImageType::RegionType   InputImageRegionType;
  typedef typename InputImageType::PixelType    InputImagePixelType;

  /** ImageDimension constants */
  itkStaticConstMacro( InputImageDimension, unsigned int,
                       TInputImage::ImageDimension );
  itkStaticConstMacro( OutputImageDimension, unsigned int,
                       TOutputImage::ImageDimension );

protected:
  GPURecursiveGaussianImageFilter();
  ~GPURecursiveGaussianImageFilter(){}
  virtual void PrintSelf( std::ostream & os, Indent indent ) const;

  virtual void GPUGenerateData();

private:
  GPURecursiveGaussianImageFilter( const Self & ); // purposely not implemented
  void operator=( const Self & );                  // purposely not implemented

  int    m_FilterGPUKernelHandle;
  size_t m_DeviceLocalMemorySize;
};

/** \class GPURecursiveGaussianImageFilterFactory
* \brief Object Factory implementation for GPURecursiveGaussianImageFilter
*/
class GPURecursiveGaussianImageFilterFactory : public ObjectFactoryBase
{
public:
  typedef GPURecursiveGaussianImageFilterFactory Self;
  typedef ObjectFactoryBase                      Superclass;
  typedef SmartPointer< Self >                   Pointer;
  typedef SmartPointer< const Self >             ConstPointer;

  /** Class methods used to interface with the registered factories. */
  virtual const char * GetITKSourceVersion() const { return ITK_SOURCE_VERSION; }
  const char * GetDescription() const { return "A Factory for GPURecursiveGaussianImageFilter"; }

  /** Method for class instantiation. */
  itkFactorylessNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPURecursiveGaussianImageFilterFactory, itk::ObjectFactoryBase );

  /** Register one factory of this type  */
  static void RegisterOneFactory( void )
  {
    GPURecursiveGaussianImageFilterFactory::Pointer factory =
      GPURecursiveGaussianImageFilterFactory::New();
    ObjectFactoryBase::RegisterFactory( factory );
  }

private:
  GPURecursiveGaussianImageFilterFactory( const Self & ); // purposely not implemented
  void operator=( const Self & );                         // purposely not implemented

#define OverrideRecursiveGaussianImageFilterTypeMacro( ipt, opt, dm1, dm2, dm3 )                               \
  {                                                                                                            \
    typedef Image< ipt, dm1 > InputImageType1D;                                                                \
    typedef Image< opt, dm1 > OutputImageType1D;                                                               \
    this->RegisterOverride(                                                                                    \
      typeid( RecursiveGaussianImageFilter< InputImageType1D, OutputImageType1D > ).name(),                    \
      typeid( GPURecursiveGaussianImageFilter< InputImageType1D, OutputImageType1D > ).name(),                 \
      "GPU RecursiveGaussianImageFilter Override 1D",                                                          \
      true,                                                                                                    \
      CreateObjectFunction< GPURecursiveGaussianImageFilter< InputImageType1D, OutputImageType1D > >::New() ); \
    typedef Image< ipt, dm2 > InputImageType2D;                                                                \
    typedef Image< opt, dm2 > OutputImageType2D;                                                               \
    this->RegisterOverride(                                                                                    \
      typeid( RecursiveGaussianImageFilter< InputImageType2D, OutputImageType2D > ).name(),                    \
      typeid( GPURecursiveGaussianImageFilter< InputImageType2D, OutputImageType2D > ).name(),                 \
      "GPU RecursiveGaussianImageFilter Override 2D",                                                          \
      true,                                                                                                    \
      CreateObjectFunction< GPURecursiveGaussianImageFilter< InputImageType2D, OutputImageType2D > >::New() ); \
    typedef Image< ipt, dm3 > InputImageType3D;                                                                \
    typedef Image< opt, dm3 > OutputImageType3D;                                                               \
    this->RegisterOverride(                                                                                    \
      typeid( RecursiveGaussianImageFilter< InputImageType3D, OutputImageType3D > ).name(),                    \
      typeid( GPURecursiveGaussianImageFilter< InputImageType3D, OutputImageType3D > ).name(),                 \
      "GPU RecursiveGaussianImageFilter Override 3D",                                                          \
      true,                                                                                                    \
      CreateObjectFunction< GPURecursiveGaussianImageFilter< InputImageType3D, OutputImageType3D > >::New() ); \
  }

  GPURecursiveGaussianImageFilterFactory()
  {
    if ( IsGPUAvailable() )
    {
      // general types
      //OverrideRecursiveGaussianImageFilterTypeMacro(unsigned char, unsigned
      // char, 1, 2, 3);
      //OverrideRecursiveGaussianImageFilterTypeMacro(char, char, 1, 2, 3);
      //OverrideRecursiveGaussianImageFilterTypeMacro(unsigned short, unsigned
      // short, 1, 2, 3);
      OverrideRecursiveGaussianImageFilterTypeMacro( short, short, 1, 2, 3 );
      //OverrideRecursiveGaussianImageFilterTypeMacro(unsigned int, unsigned
      // int, 1, 2, 3);
      //OverrideRecursiveGaussianImageFilterTypeMacro(int, int, 1, 2, 3);
      OverrideRecursiveGaussianImageFilterTypeMacro( float, float, 1, 2, 3 );
      //OverrideRecursiveGaussianImageFilterTypeMacro(double, double, 1, 2, 3);

      // type to float, float to type
      //OverrideRecursiveGaussianImageFilterTypeMacro(unsigned char, float, 1,
      // 2, 3);
      //OverrideRecursiveGaussianImageFilterTypeMacro(float, unsigned char, 1,
      // 2, 3);
      //OverrideRecursiveGaussianImageFilterTypeMacro(char, float, 1, 2, 3);
      //OverrideRecursiveGaussianImageFilterTypeMacro(float, char, 1, 2, 3);
      //OverrideRecursiveGaussianImageFilterTypeMacro(unsigned short, float, 1,
      // 2, 3);
      //OverrideRecursiveGaussianImageFilterTypeMacro(float, unsigned short, 1,
      // 2, 3);
      OverrideRecursiveGaussianImageFilterTypeMacro( short, float, 1, 2, 3 );
      //OverrideRecursiveGaussianImageFilterTypeMacro(float, short, 1, 2, 3);
      //OverrideRecursiveGaussianImageFilterTypeMacro(unsigned int, float, 1, 2,
      // 3);
      //OverrideRecursiveGaussianImageFilterTypeMacro(float, unsigned int, 1, 2,
      // 3);
      //OverrideRecursiveGaussianImageFilterTypeMacro(int, float, 1, 2, 3);
      //OverrideRecursiveGaussianImageFilterTypeMacro(float, int, 1, 2, 3);
      //OverrideRecursiveGaussianImageFilterTypeMacro(float, double, 1, 2, 3);
      //OverrideRecursiveGaussianImageFilterTypeMacro(double, float, 1, 2, 3);
    }
  }
};
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGPURecursiveGaussianImageFilter.hxx"
#endif

#endif /* __itkGPURecursiveGaussianImageFilter_h */
