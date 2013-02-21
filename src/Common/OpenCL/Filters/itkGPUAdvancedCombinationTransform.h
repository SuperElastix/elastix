/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUAdvancedCombinationTransform_h
#define __itkGPUAdvancedCombinationTransform_h

#include "itkAdvancedCombinationTransform.h"
#include "itkGPUCompositeTransformBase.h"

namespace itk
{
/** \class GPUAdvancedCombinationTransform
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
          class TParentImageFilter = AdvancedCombinationTransform< TScalarType, NDimensions > >
class GPUAdvancedCombinationTransform :
  public TParentImageFilter, public GPUCompositeTransformBase< TScalarType, NDimensions >
{
public:
  /** Standard class typedefs. */
  typedef GPUAdvancedCombinationTransform Self;
  typedef TParentImageFilter              CPUSuperclass;
  typedef GPUCompositeTransformBase       GPUSuperclass;
  typedef SmartPointer< Self >            Pointer;
  typedef SmartPointer< const Self >      ConstPointer;

  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUAdvancedCombinationTransform, TParentImageFilter );

  /** Sub transform type */
  typedef typename GPUSuperclass::TransformType        GPUTransformType;
  typedef typename GPUSuperclass::TransformTypePointer TransformTypePointer;

  /** Typedefs for the InitialTransform. */
  typedef typename CPUSuperclass::InitialTransformType         InitialTransformType;
  typedef typename CPUSuperclass::InitialTransformPointer      InitialTransformPointer;
  typedef typename CPUSuperclass::InitialTransformConstPointer InitialTransformConstPointer;

  /** Typedefs for the CurrentTransform. */
  typedef typename CPUSuperclass::CurrentTransformType         CurrentTransformType;
  typedef typename CPUSuperclass::CurrentTransformPointer      CurrentTransformPointer;
  typedef typename CPUSuperclass::CurrentTransformConstPointer CurrentTransformConstPointer;

  /** Get number of transforms in composite transform. */
  virtual size_t GetNumberOfTransforms() const;

  /** Get the Nth transform. */
  virtual TransformTypePointer GetNthTransform( SizeValueType n );

  /** Get the Nth transform, const version. */
  virtual TransformTypeConstPointer GetNthTransform( SizeValueType n ) const;

protected:
  GPUAdvancedCombinationTransform() {}
  virtual ~GPUAdvancedCombinationTransform() {}
  void PrintSelf( std::ostream & s, Indent indent ) const;

private:
  GPUAdvancedCombinationTransform( const Self & other ); // purposely not implemented
  const Self & operator=( const Self & );                // purposely not implemented
};

/** \class GPUAdvancedCombinationTransformFactory
* \brief Object Factory implementation for GPUAdvancedCombinationTransform
*/
class GPUAdvancedCombinationTransformFactory : public ObjectFactoryBase
{
public:
  typedef GPUAdvancedCombinationTransformFactory Self;
  typedef ObjectFactoryBase                      Superclass;
  typedef SmartPointer< Self >                   Pointer;
  typedef SmartPointer< const Self >             ConstPointer;

  /** Class methods used to interface with the registered factories. */
  virtual const char * GetITKSourceVersion() const { return ITK_SOURCE_VERSION; }
  const char * GetDescription() const { return "A Factory for GPUAdvancedCombinationTransform"; }

  /** Method for class instantiation. */
  itkFactorylessNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUAdvancedCombinationTransformFactory, itk::ObjectFactoryBase );

  /** Register one factory of this type  */
  static void RegisterOneFactory( void )
  {
    GPUAdvancedCombinationTransformFactory::Pointer factory =
      GPUAdvancedCombinationTransformFactory::New();
    ObjectFactoryBase::RegisterFactory( factory );
  }

private:
  GPUAdvancedCombinationTransformFactory( const Self & ); // purposely not implemented
  void operator=( const Self & );                         // purposely not implemented

#define OverrideAdvancedCombinationTransformTypeMacro( st, dm )                   \
  {                                                                               \
    this->RegisterOverride(                                                       \
      typeid( AdvancedCombinationTransform< st, dm > ).name(),                    \
      typeid( GPUAdvancedCombinationTransform< st, dm > ).name(),                 \
      "GPU AdvancedCombinationTransform Override",                                \
      true,                                                                       \
      CreateObjectFunction< GPUAdvancedCombinationTransform< st, dm > >::New() ); \
  }

  GPUAdvancedCombinationTransformFactory()
  {
    if ( IsGPUAvailable() )
    {
      OverrideAdvancedCombinationTransformTypeMacro( float, 1 );
      //OverrideAdvancedCombinationTransformTypeMacro(double, 1);

      OverrideAdvancedCombinationTransformTypeMacro( float, 2 );
      //OverrideAdvancedCombinationTransformTypeMacro(double, 2);

      OverrideAdvancedCombinationTransformTypeMacro( float, 3 );
      //OverrideAdvancedCombinationTransformTypeMacro(double, 3);
    }
  }
};
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGPUAdvancedCombinationTransform.hxx"
#endif

#endif /* __itkGPUAdvancedCombinationTransform_h */
