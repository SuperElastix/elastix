/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUCompositeTransform_h
#define __itkGPUCompositeTransform_h

#include "itkCompositeTransform.h"
#include "itkGPUCompositeTransformBase.h"

namespace itk
{
/** \class GPUCompositeTransform
* \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
* Department of Radiology, Leiden, The Netherlands
*
* This implementation was taken from elastix (http://elastix.isi.uu.nl/).
*
* \note This work was funded by the Netherlands Organisation for
* Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
*
*/
template< class TScalarType = float, unsigned int NDimensions = 3,
          class TParentImageFilter = CompositeTransform< TScalarType, NDimensions > >
class GPUCompositeTransform :
  public TParentImageFilter, public GPUCompositeTransformBase< TScalarType, NDimensions >
{
public:
  /** Standard class typedefs. */
  typedef GPUCompositeTransform                                 Self;
  typedef TParentImageFilter                                    CPUSuperclass;
  typedef GPUCompositeTransformBase< TScalarType, NDimensions > GPUSuperclass;
  typedef SmartPointer< Self >                                  Pointer;
  typedef SmartPointer< const Self >                            ConstPointer;

  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUCompositeTransform, TParentImageFilter );

  /** Sub transform type */
  typedef typename GPUSuperclass::TransformType             GPUTransformType;
  typedef typename GPUSuperclass::TransformTypePointer      TransformTypePointer;
  typedef typename GPUSuperclass::TransformTypeConstPointer TransformTypeConstPointer;

  /** Get number of transforms in composite transform. */
  virtual SizeValueType GetNumberOfTransforms() const;
  { return CPUSuperclass::GetNumberOfTransforms(); }

  /** Get the Nth transform. */
  virtual TransformTypePointer GetNthTransform( SizeValueType n )
  { return CPUSuperclass::GetNthTransform( n ); }

  /** Get the Nth transform, const version. */
  virtual TransformTypeConstPointer GetNthTransform( SizeValueType n ) const
  { return CPUSuperclass::GetNthTransform( n ); }

protected:
  GPUCompositeTransform() {}
  virtual ~GPUCompositeTransform() {}
  void PrintSelf( std::ostream & s, Indent indent ) const
  { CPUSuperclass::PrintSelf( s, indent ); }

private:
  GPUCompositeTransform( const Self & other ); // purposely not implemented
  const Self & operator=( const Self & );      // purposely not implemented
}

/** \class GPUCompositeTransformFactory
* \brief Object Factory implementation for GPUCompositeTransform
*/
class GPUCompositeTransformFactory : public ObjectFactoryBase
{
public:
  typedef GPUCompositeTransformFactory Self;
  typedef ObjectFactoryBase            Superclass;
  typedef SmartPointer< Self >         Pointer;
  typedef SmartPointer< const Self >   ConstPointer;

  /** Class methods used to interface with the registered factories. */
  virtual const char * GetITKSourceVersion() const { return ITK_SOURCE_VERSION; }
  const char * GetDescription() const { return "A Factory for GPUCompositeTransform"; }

  /** Method for class instantiation. */
  itkFactorylessNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUCompositeTransformFactory, itk::ObjectFactoryBase );

  /** Register one factory of this type  */
  static void RegisterOneFactory( void )
  {
    GPUCompositeTransformFactory::Pointer factory =
      GPUCompositeTransformFactory::New();
    ObjectFactoryBase::RegisterFactory( factory );
  }

private:
  GPUCompositeTransformFactory( const Self & );   // purposely not implemented
  void operator=( const Self & );                 // purposely not implemented

#define OverrideCompositeTransformTypeMacro( st, dm )                   \
  {                                                                     \
    this->RegisterOverride(                                             \
      typeid( CompositeTransform< st, dm > ).name(),                    \
      typeid( GPUCompositeTransform< st, dm > ).name(),                 \
      "GPU CompositeTransform Override",                                \
      true,                                                             \
      CreateObjectFunction< GPUCompositeTransform< st, dm > >::New() ); \
  }

  GPUCompositeTransformFactory()
  {
    if ( IsGPUAvailable() )
    {
      OverrideCompositeTransformTypeMacro( float, 1 );
      //OverrideCompositeTransformTypeMacro(double, 1);

      OverrideCompositeTransformTypeMacro( float, 2 );
      //OverrideCompositeTransformTypeMacro(double, 2);

      OverrideCompositeTransformTypeMacro( float, 3 );
      //OverrideCompositeTransformTypeMacro(double, 3);
    }
  }
};
} // end namespace itk

#endif /* __itkGPUCompositeTransform_h */
