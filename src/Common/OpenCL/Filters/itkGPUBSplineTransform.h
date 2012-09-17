/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUBSplineTransform_h
#define __itkGPUBSplineTransform_h

#include "itkObjectFactoryBase.h"
#include "itkVersion.h"

#include "itkBSplineTransform.h"
#include "itkGPUBSplineBaseTransform.h"

namespace itk
{
/** \class GPUBSplineTransform
 */

/** Create a helper GPU Kernel class for GPUBSplineTransform */
itkGPUKernelClassMacro( GPUBSplineTransformKernel );

template< class TScalarType = float, unsigned int NDimensions = 3, unsigned int VSplineOrder = 3,
          class TParentImageFilter = BSplineTransform< TScalarType, NDimensions, VSplineOrder > >
class GPUBSplineTransform :
  public TParentImageFilter, public GPUBSplineBaseTransform< TScalarType, NDimensions >
{
public:
  /** Standard class typedefs. */
  typedef GPUBSplineTransform                                 Self;
  typedef TParentImageFilter                                  Superclass;
  typedef GPUBSplineBaseTransform< TScalarType, NDimensions > SuperSuperclass;
  typedef SmartPointer< Self >                                Pointer;
  typedef SmartPointer< const Self >                          ConstPointer;
  typedef typename Superclass::ParametersType                 ParametersType;
  typedef typename Superclass::CoefficientImageArray          CoefficientImageArray;

  itkNewMacro( Self );

  /** Dimension of the domain space. */
  itkStaticConstMacro( SpaceDimension, unsigned int, NDimensions );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUBSplineTransform, TParentImageFilter );

  void SetParameters( const ParametersType & parameters );
  void SetCoefficientImages( const CoefficientImageArray & images );

protected:
  GPUBSplineTransform();
  virtual ~GPUBSplineTransform() {}

  void PrintSelf( std::ostream & s, Indent indent ) const;

  virtual bool GetSourceCode( std::string & _source ) const;

  void CopyCoefficientImagesToGPU();

private:
  GPUBSplineTransform( const Self & other ); // purposely not implemented
  const Self & operator=( const Self & );    // purposely not implemented

  std::vector< std::string > m_Sources;
  bool                       m_SourcesLoaded;
};

/** \class GPUBSplineTransformFactory
* \brief Object Factory implementation for GPUBSplineTransform
*/
class GPUBSplineTransformFactory : public ObjectFactoryBase
{
public:
  typedef GPUBSplineTransformFactory Self;
  typedef ObjectFactoryBase          Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** Class methods used to interface with the registered factories. */
  virtual const char * GetITKSourceVersion() const { return ITK_SOURCE_VERSION; }
  const char * GetDescription() const { return "A Factory for GPUBSplineTransform"; }

  /** Method for class instantiation. */
  itkFactorylessNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUBSplineTransformFactory, itk::ObjectFactoryBase );

  /** Register one factory of this type  */
  static void RegisterOneFactory( void )
  {
    GPUBSplineTransformFactory::Pointer factory =
      GPUBSplineTransformFactory::New();
    ObjectFactoryBase::RegisterFactory( factory );
  }

private:
  GPUBSplineTransformFactory( const Self & ); // purposely not implemented
  void operator=( const Self & );             // purposely not implemented

#define OverrideBSplineTransformTypeMacro( st, dm, so )                   \
  {                                                                       \
    this->RegisterOverride(                                               \
      typeid( BSplineTransform< st, dm, so > ).name(),                    \
      typeid( GPUBSplineTransform< st, dm, so > ).name(),                 \
      "GPU BSplineTransform Override",                                    \
      true,                                                               \
      CreateObjectFunction< GPUBSplineTransform< st, dm, so > >::New() ); \
  }

  GPUBSplineTransformFactory()
  {
    if ( IsGPUAvailable() )
    {
      // SplineOrder = 1
      OverrideBSplineTransformTypeMacro( float, 1, 1 );
      //OverrideBSplineTransformTypeMacro(double, 1, 1);

      OverrideBSplineTransformTypeMacro( float, 2, 1 );
      //OverrideBSplineTransformTypeMacro(double, 2, 1);

      OverrideBSplineTransformTypeMacro( float, 3, 1 );
      //OverrideBSplineTransformTypeMacro(double, 3, 1);

      // SplineOrder = 2
      OverrideBSplineTransformTypeMacro( float, 1, 2 );
      //OverrideBSplineTransformTypeMacro(double, 1, 2);

      OverrideBSplineTransformTypeMacro( float, 2, 2 );
      //OverrideBSplineTransformTypeMacro(double, 2, 2);

      OverrideBSplineTransformTypeMacro( float, 3, 2 );
      //OverrideBSplineTransformTypeMacro(double, 3, 2);

      // SplineOrder = 3
      OverrideBSplineTransformTypeMacro( float, 1, 3 );
      //OverrideBSplineTransformTypeMacro(double, 1, 3);

      OverrideBSplineTransformTypeMacro( float, 2, 3 );
      //OverrideBSplineTransformTypeMacro(double, 2, 3);

      OverrideBSplineTransformTypeMacro( float, 3, 3 );
      //OverrideBSplineTransformTypeMacro(double, 3, 3);
    }
  }
};
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGPUBSplineTransform.hxx"
#endif

#endif /* __itkGPUBSplineTransform_h */
