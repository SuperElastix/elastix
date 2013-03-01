/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUAdvancedBSplineDeformableTransform_h
#define __itkGPUAdvancedBSplineDeformableTransform_h

#include "itkObjectFactoryBase.h"
#include "itkVersion.h"

#include "itkAdvancedBSplineDeformableTransform.h"
#include "itkGPUBSplineBaseTransform.h"

namespace itk
{
/** \class GPUAdvancedBSplineDeformableTransform
* \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
* Department of Radiology, Leiden, The Netherlands
*
* This implementation was taken from elastix (http://elastix.isi.uu.nl/).
*
* \note This work was funded by the Netherlands Organisation for
* Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
*
*/
template< class TScalarType = float, unsigned int NDimensions = 3, unsigned int VSplineOrder = 3,
          class TParentImageFilter = AdvancedBSplineDeformableTransform< TScalarType, NDimensions, VSplineOrder > >
class GPUAdvancedBSplineDeformableTransform :
  public TParentImageFilter, public GPUBSplineBaseTransform< TScalarType, NDimensions, VSplineOrder >
{
public:
  /** Standard class typedefs. */
  typedef GPUAdvancedBSplineDeformableTransform               Self;
  typedef TParentImageFilter                                  CPUSuperclass;
  typedef GPUBSplineBaseTransform< TScalarType, NDimensions > GPUSuperclass;
  typedef SmartPointer< Self >                                Pointer;
  typedef SmartPointer< const Self >                          ConstPointer;
  typedef typename CPUSuperclass::ParametersType              ParametersType;
  typedef typename CPUSuperclass::ImagePointer                ImagePointer;

  itkNewMacro( Self );

  /** Dimension of the domain space. */
  itkStaticConstMacro( SpaceDimension, unsigned int, NDimensions );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUAdvancedBSplineDeformableTransform, TParentImageFilter );

  void SetParameters( const ParametersType & parameters );

  void SetCoefficientImages( ImagePointer images[] );

protected:
  GPUAdvancedBSplineDeformableTransform() {}
  virtual ~GPUAdvancedBSplineDeformableTransform() {}

  void PrintSelf( std::ostream & s, Indent indent ) const;

  void CopyCoefficientImagesToGPU();

private:
  GPUAdvancedBSplineDeformableTransform( const Self & other ); // purposely not implemented
  const Self & operator=( const Self & );                      // purposely not implemented
};

/** \class GPUAdvancedBSplineDeformableTransformFactory
* \brief Object Factory implementation for GPUAdvancedBSplineDeformableTransform
*/
class GPUAdvancedBSplineDeformableTransformFactory : public ObjectFactoryBase
{
public:
  typedef GPUAdvancedBSplineDeformableTransformFactory Self;
  typedef ObjectFactoryBase                            Superclass;
  typedef SmartPointer< Self >                         Pointer;
  typedef SmartPointer< const Self >                   ConstPointer;

  /** Class methods used to interface with the registered factories. */
  virtual const char * GetITKSourceVersion() const { return ITK_SOURCE_VERSION; }
  const char * GetDescription() const { return "A Factory for GPUAdvancedBSplineDeformableTransform"; }

  /** Method for class instantiation. */
  itkFactorylessNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUAdvancedBSplineDeformableTransformFactory, itk::ObjectFactoryBase );

  /** Register one factory of this type  */
  static void RegisterOneFactory( void )
  {
    GPUAdvancedBSplineDeformableTransformFactory::Pointer factory =
      GPUAdvancedBSplineDeformableTransformFactory::New();
    ObjectFactoryBase::RegisterFactory( factory );
  }

private:
  GPUAdvancedBSplineDeformableTransformFactory( const Self & ); // purposely not implemented
  void operator=( const Self & );                               // purposely not implemented

#define OverrideAdvancedBSplineDeformableTransformTypeMacro( st, dm, so )                   \
  {                                                                                         \
    this->RegisterOverride(                                                                 \
      typeid( AdvancedBSplineDeformableTransform< st, dm, so > ).name(),                    \
      typeid( GPUAdvancedBSplineDeformableTransform< st, dm, so > ).name(),                 \
      "GPU AdvancedBSplineDeformableTransform Override",                                    \
      true,                                                                                 \
      CreateObjectFunction< GPUAdvancedBSplineDeformableTransform< st, dm, so > >::New() ); \
  }

  GPUAdvancedBSplineDeformableTransformFactory()
  {
    if ( IsGPUAvailable() )
    {
      // SplineOrder = 1
      OverrideAdvancedBSplineDeformableTransformTypeMacro( float, 1, 1 );
      //OverrideAdvancedBSplineDeformableTransformTypeMacro(double, 1, 1);

      OverrideAdvancedBSplineDeformableTransformTypeMacro( float, 2, 1 );
      //OverrideAdvancedBSplineDeformableTransformTypeMacro(double, 2, 1);

      OverrideAdvancedBSplineDeformableTransformTypeMacro( float, 3, 1 );
      //OverrideAdvancedBSplineDeformableTransformTypeMacro(double, 3, 1);

      // SplineOrder = 2
      OverrideAdvancedBSplineDeformableTransformTypeMacro( float, 1, 2 );
      //OverrideAdvancedBSplineDeformableTransformTypeMacro(double, 1, 2);

      OverrideAdvancedBSplineDeformableTransformTypeMacro( float, 2, 2 );
      //OverrideAdvancedBSplineDeformableTransformTypeMacro(double, 2, 2);

      OverrideAdvancedBSplineDeformableTransformTypeMacro( float, 3, 2 );
      //OverrideAdvancedBSplineDeformableTransformTypeMacro(double, 3, 2);

      // SplineOrder = 3
      OverrideAdvancedBSplineDeformableTransformTypeMacro( float, 1, 3 );
      //OverrideAdvancedBSplineDeformableTransformTypeMacro(double, 1, 3);

      OverrideAdvancedBSplineDeformableTransformTypeMacro( float, 2, 3 );
      //OverrideAdvancedBSplineDeformableTransformTypeMacro(double, 2, 3);

      OverrideAdvancedBSplineDeformableTransformTypeMacro( float, 3, 3 );
      //OverrideAdvancedBSplineDeformableTransformTypeMacro(double, 3, 3);
    }
  }
};
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGPUAdvancedBSplineDeformableTransform.hxx"
#endif

#endif /* __itkGPUAdvancedBSplineDeformableTransform_h */
